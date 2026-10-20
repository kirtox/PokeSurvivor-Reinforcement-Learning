"""Gymnasium environment wrapping the local PokeSurvivor JS game via Playwright.

Scope: fixed partner (Ditto, dex 132). The agent controls movement, active-move
switching and the level-up card choice (third action dim, used only while a card
choice is pending). Items / bonds are auto-picked inside the page
(see game/js/rl_boot.js).
"""
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from playwright.sync_api import sync_playwright

GAME_HTML = (Path(__file__).resolve().parent.parent / "game" / "rl_harness.html").as_uri()

MAX_ENEMIES = 8
MAX_HAZARDS = 6
MAX_PICKUPS = 6
RANGE_NORM = 600.0          # px used to squash dx/dy into roughly [-1, 1]
MAX_TIME = 30 * 60.0        # seconds; used only to normalize the time feature
MAX_SLOTS = 4
MAX_CARDS = 3
N_RAYS = 8                  # free distance toward each of the 8 move directions (walls/props/harmful liquid)
STUCK_PENALTY = 0.05        # per step that asked to move but barely moved

# Must cover every `kind` in game/js/data/moves.js; unknown kinds are left all-zero.
MOVE_KINDS = ['projectile', 'melee', 'beam', 'orbit', 'nova', 'aura', 'buff', 'boomerang',
              'chain', 'mine', 'meteor', 'turret', 'dash', 'trail', 'cone']
CARD_TYPES = ['new', 'up', 'evo', 'stat']
STAT_IDS = ['hp', 'atk', 'spd', 'cd', 'def', 'rgn', 'mag', 'xp', 'heal']  # see game/js/systems/levelup.js

# slot: present + kind one-hot + level frac + group size + ready + dmg + cd
SLOT_DIM = 1 + len(MOVE_KINDS) + 5
# card: present + type one-hot + stat one-hot + evoStat + kind one-hot + level frac + dmg + cd
CARD_DIM = 1 + len(CARD_TYPES) + len(STAT_IDS) + 1 + len(MOVE_KINDS) + 3

OBS_DIM = (4 + 4 + 1 + MAX_ENEMIES * 5 + MAX_HAZARDS * 5 + MAX_PICKUPS * 4
           + 2 + MAX_SLOTS * SLOT_DIM + MAX_CARDS * CARD_DIM
           + N_RAYS + 1)

PARTNER_DEX = 150           # Mewtwo (was 132 Ditto in v1/v2/v3 trials)
PARTNER_SHINY = True        # shiny skin only; needs the shiny sprite to exist for that dex
# Permanent per-Pokemon upgrades (ids from game/js/data/upgrades.js), applied at every reset.
# Maxed combat/survival stats; moves=1 -> start with 2 moves. reroll/ban stay 0: the agent has no
# interface for them. coins is irrelevant for training.
PARTNER_UPGRADES = {"hp": 10, "atk": 10, "spd": 8, "cd": 8, "def": 8, "regen": 5,
                    "magnet": 6, "xp": 8, "crit": 10, "moves": 1}


def _encode(obs: dict) -> np.ndarray:
    v = np.zeros(OBS_DIM, dtype=np.float32)
    if obs is None:
        return v
    i = 0
    v[i] = obs["hp"] / max(1.0, obs["maxHp"]); i += 1
    v[i] = min(1.0, obs["level"] / 50.0); i += 1
    v[i] = obs["xp"] / max(1.0, obs["xpNext"]); i += 1
    v[i] = min(1.0, obs["time"] / MAX_TIME); i += 1
    active = obs["active"]
    for slot in range(4):
        v[i] = 1.0 if slot == active else 0.0; i += 1
    v[i] = min(1.0, obs["kills"] / 200.0); i += 1
    for e in obs["enemies"][:MAX_ENEMIES]:
        v[i] = e["dx"] / RANGE_NORM; i += 1
        v[i] = e["dy"] / RANGE_NORM; i += 1
        v[i] = e["hp"]; i += 1
        v[i] = 1.0 if e["boss"] else 0.0; i += 1
        v[i] = 1.0; i += 1  # presence flag
    i += (MAX_ENEMIES - len(obs["enemies"])) * 5
    for h in obs["hazards"][:MAX_HAZARDS]:
        v[i] = h["dx"] / RANGE_NORM; i += 1
        v[i] = h["dy"] / RANGE_NORM; i += 1
        v[i] = (h["r"] or 40.0) / RANGE_NORM; i += 1
        v[i] = h["frac"]; i += 1
        v[i] = 1.0; i += 1
    i += (MAX_HAZARDS - len(obs["hazards"])) * 5
    for p in obs.get("pickups", [])[:MAX_PICKUPS]:
        v[i] = p["dx"] / RANGE_NORM; i += 1
        v[i] = p["dy"] / RANGE_NORM; i += 1
        v[i] = p["kind"] / 4.0; i += 1   # 1=xp 2=xpBig 3=heal 4=magnet
        v[i] = 1.0; i += 1               # presence flag
    i += (MAX_PICKUPS - len(obs.get("pickups", [])[:MAX_PICKUPS])) * 4

    # free move slots + "a card choice is pending" flag
    v[i] = max(0, obs["maxMoves"] - obs["numSlots"]) / 4.0; i += 1
    v[i] = 1.0 if obs["cards"] else 0.0; i += 1

    base = i
    for s, m in enumerate(obs["moves"][:MAX_SLOTS]):
        j = base + s * SLOT_DIM
        v[j] = 1.0; j += 1
        if m["kind"] in MOVE_KINDS:
            v[j + MOVE_KINDS.index(m["kind"])] = 1.0
        j += len(MOVE_KINDS)
        v[j] = m["lvl"]
        v[j + 1] = min(1.0, m["size"] / 4.0)
        v[j + 2] = m["ready"]
        v[j + 3] = min(1.5, m["dmg"] / 60.0)
        v[j + 4] = min(1.5, m["cd"] / 3.0)
    base += MAX_SLOTS * SLOT_DIM

    for c, card in enumerate(obs["cards"][:MAX_CARDS]):
        j = base + c * CARD_DIM
        v[j] = 1.0; j += 1
        if card["type"] in CARD_TYPES:
            v[j + CARD_TYPES.index(card["type"])] = 1.0
        j += len(CARD_TYPES)
        if card["stat"] in STAT_IDS:
            v[j + STAT_IDS.index(card["stat"])] = 1.0
        j += len(STAT_IDS)
        v[j] = card["evoStat"]; j += 1
        if card["kind"] in MOVE_KINDS:
            v[j + MOVE_KINDS.index(card["kind"])] = 1.0
        j += len(MOVE_KINDS)
        v[j] = card["lvl"]
        v[j + 1] = min(1.5, card["dmg"] / 60.0)
        v[j + 2] = min(1.5, card["cd"] / 3.0)

    base += MAX_CARDS * CARD_DIM
    rmax = float(obs.get("rayMax", 192.0))
    for k, d in enumerate(obs.get("rays", [])[:N_RAYS]):
        v[base + k] = d / rmax            # 0 = blocked right next to us, 1 = clear
    v[base + N_RAYS] = 1.0 if obs.get("stuck") else 0.0
    return v


class PokeSurvivorEnv(gym.Env):
    """One browser tab = one environment instance."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, headless: bool = True, speed: float = 10.0, frame_skip: int = 4,
                 max_time: float = MAX_TIME, partner_dex: int = PARTNER_DEX,
                 partner_shiny: bool = PARTNER_SHINY, upgrades: dict | None = None):
        super().__init__()
        self.headless = headless
        self.speed = speed
        self.frame_skip = frame_skip
        self.max_time = max_time
        self.partner_dex = partner_dex
        self.partner_shiny = partner_shiny
        self.upgrades = dict(PARTNER_UPGRADES if upgrades is None else upgrades)

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)
        # [direction (0=stay,1-8), skill (0=keep,1-4=slot), card (0-2, only used while a level-up is pending)]
        self.action_space = spaces.MultiDiscrete([9, 5, MAX_CARDS])

        self._pw = None
        self._browser = None
        self._page = None
        self._prev_obs = None

    def _ensure_page(self):
        if self._page is not None:
            return
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()
        self._page.goto(GAME_HTML)
        self._page.wait_for_function("() => !!window.RL")
        self._page.evaluate("(k) => window.RL.setSpeed(k)", self.speed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_page()
        raw = self._page.evaluate("([dex, sh, up]) => window.RL.reset(dex, sh, up)",
                                  [self.partner_dex, self.partner_shiny, self.upgrades])
        self._page.evaluate("(k) => window.RL.setSpeed(k)", self.speed)
        self._prev_obs = raw
        return _encode(raw), {}

    def step(self, action):
        move_idx, skill_idx, card_idx = int(action[0]), int(action[1]), int(action[2])
        raw = self._page.evaluate(
            "(a) => window.RL.step(a[0], a[1], a[2], a[3])",
            [move_idx, skill_idx, self.frame_skip, card_idx],
        )
        reward, terminated = self._reward(self._prev_obs, raw)
        truncated = (not terminated) and raw["time"] >= self.max_time
        self._prev_obs = raw
        return _encode(raw), reward, terminated, truncated, {"raw": raw}

    @staticmethod
    def _reward(prev, cur):
        if prev is None:
            return 0.0, False
        dead = bool(cur["dead"])
        r = 0.02  # small per-step survival reward
        # Smooth level progress (level + fraction of the bar) so collecting XP is rewarded
        # immediately, not only on level-up. 12.0 per full level (was 5.0: the agent levelled
        # too slowly, ~3 levels in 57s).
        prog_prev = prev["level"] + prev["xp"] / max(1.0, prev["xpNext"])
        prog_cur = cur["level"] + cur["xp"] / max(1.0, cur["xpNext"])
        r += max(0.0, prog_cur - prog_prev) * 12.0
        r += max(0, cur["kills"] - prev["kills"]) * 1.0
        hp_loss = max(0.0, prev["hp"] / max(1.0, prev["maxHp"]) - cur["hp"] / max(1.0, cur["maxHp"]))
        r -= hp_loss * 5.0
        if cur.get("stuck"):
            r -= STUCK_PENALTY
        if dead:
            r -= 10.0
        return r, dead

    def render(self):
        pass  # Window A (headed Playwright) *is* the render; nothing to do here.

    def close(self):
        if self._browser is not None:
            self._browser.close()
        if self._pw is not None:
            self._pw.stop()
        self._page = self._browser = self._pw = None
