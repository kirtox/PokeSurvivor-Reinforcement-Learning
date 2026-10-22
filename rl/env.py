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
MAX_SHOTS = 8               # enemy projectiles in flight (nearest first)
MAX_TRAPS = 4               # ground traps (poison / sticky / explosive)
MAX_PICKUPS = 6
SHOT_SPEED_NORM = 200.0     # enemy shots fly at ~190 px/s
BEAM_LEN_NORM = 400.0
HAZARD_TIME_NORM = 2.0      # seconds until a warned hazard detonates
ENEMY_DIM = 24              # dx dy hp boss present flying line-of-fire-clear is-game-target | vx vy | behavior one-hot
                            # | aim dash fuse jump | phase-time-left fury shield
HAZARD_DIM = 12             # dx dy gap r frac time-left beam cos sin len inside present
SHOT_DIM = 5                # dx dy vx vy present
TRAP_DIM = 4                # dx dy kind present
RANGE_NORM = 600.0          # px used to squash dx/dy into roughly [-1, 1]
MAX_TIME = 30 * 60.0        # seconds; used only to normalize the time feature
MAX_SLOTS = 4
MAX_CARDS = 3
N_RAYS = 8                  # free distance toward each of the 8 move directions (walls/props/harmful liquid)
ENEMY_BEHAVIORS = ['charger', 'ranged', 'bomber', 'shielder', 'jumper', 'thief', 'runner']
POWERS = 4                  # altar powers: dmg spd mag cd
BUFFS = 6                   # atk patk satk spd def regen
LIQUID_KINDS = ['water', 'swamp', 'lava', 'poison', 'spark', 'ice']
WEATHER_KINDS = ['sun', 'rain', 'sand', 'snow', 'fog']
SPECIAL_TYPES = ['spring', 'altar', 'bigchest']
MAX_SPECIALS = 4
SPECIAL_DIM = 8             # dx dy type one-hot(3) state kind present
SELF_DIM = 3 + 2 + (POWERS + 1) + BUFFS + 4 + len(LIQUID_KINDS)   # vel spd | slow invuln | power | buffs | liquid state | liquid kind
WEATHER_DIM = len(WEATHER_KINDS) + 2
DENSITY_DIM = 8 + 1         # enemies per move-direction sector + total alive
RIFT_DIM = 10               # rift present dx dy ttl | arena present time-left cdx cdy edge won
N_MOVES = 9                 # previous move action one-hot
ENEMY_VEL_NORM = 150.0
PLAYER_VEL_NORM = 250.0
PHASE_TIME_NORM = 1.5
STUCK_PENALTY = 0.05        # per step that asked to move but barely moved
TRAP_PENALTY = 0.5          # per ground trap that fires on us (sticky traps cost no HP, so HP loss alone misses them)
AIM_BLOCKED_PENALTY = 0.02  # per step with a projectile/beam move whose auto-target is behind cover

# Must cover every `kind` in game/js/data/moves.js; unknown kinds are left all-zero.
MOVE_KINDS = ['projectile', 'melee', 'beam', 'orbit', 'nova', 'aura', 'buff', 'boomerang',
              'chain', 'mine', 'meteor', 'turret', 'dash', 'trail', 'cone']
CARD_TYPES = ['new', 'up', 'evo', 'stat']
STAT_IDS = ['hp', 'atk', 'spd', 'cd', 'def', 'rgn', 'mag', 'xp', 'heal']  # see game/js/systems/levelup.js

# slot: present + kind one-hot + level frac + group size + ready + dmg + cd
SLOT_DIM = 1 + len(MOVE_KINDS) + 5
# card: present + type one-hot + stat one-hot + evoStat + kind one-hot + level frac + dmg + cd
CARD_DIM = 1 + len(CARD_TYPES) + len(STAT_IDS) + 1 + len(MOVE_KINDS) + 3

OBS_DIM = (4 + 4 + 1 + MAX_ENEMIES * ENEMY_DIM + MAX_HAZARDS * HAZARD_DIM
           + MAX_SHOTS * SHOT_DIM + MAX_TRAPS * TRAP_DIM + MAX_PICKUPS * 4
           + 2 + MAX_SLOTS * SLOT_DIM + MAX_CARDS * CARD_DIM
           + N_RAYS + 2
           + SELF_DIM + N_MOVES + MAX_SPECIALS * SPECIAL_DIM + N_RAYS + WEATHER_DIM + DENSITY_DIM + RIFT_DIM)

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
        v[i] = 1.0 if e.get("fly") else 0.0; i += 1
        v[i] = 1.0 if e.get("los", 1) else 0.0; i += 1   # 0 = a shot toward it hits a wall/tree/rock
        v[i] = 1.0 if e.get("tgt") else 0.0; i += 1      # the enemy the game is auto-aiming at
        v[i] = e.get("vx", 0.0) / ENEMY_VEL_NORM; i += 1
        v[i] = e.get("vy", 0.0) / ENEMY_VEL_NORM; i += 1
        beh = e.get("beh", -1)
        if 0 <= beh < len(ENEMY_BEHAVIORS):
            v[i + beh] = 1.0
        i += len(ENEMY_BEHAVIORS)
        v[i] = float(e.get("aim", 0)); i += 1
        v[i] = float(e.get("dash", 0)); i += 1
        v[i] = float(e.get("fuse", 0)); i += 1
        v[i] = float(e.get("jump", 0)); i += 1
        v[i] = min(1.0, e.get("left", 0.0) / PHASE_TIME_NORM); i += 1
        v[i] = float(e.get("fury", 0)); i += 1
        v[i] = float(e.get("shield", 0)); i += 1
    i += (MAX_ENEMIES - len(obs["enemies"][:MAX_ENEMIES])) * ENEMY_DIM
    for h in obs["hazards"][:MAX_HAZARDS]:
        v[i] = h["dx"] / RANGE_NORM; i += 1
        v[i] = h["dy"] / RANGE_NORM; i += 1
        v[i] = h.get("gap", 0.0) / RANGE_NORM; i += 1     # 0 = we are inside the danger area
        v[i] = (h["r"] or 40.0) / RANGE_NORM; i += 1
        v[i] = h["frac"]; i += 1
        v[i] = min(1.0, h.get("left", 0.0) / HAZARD_TIME_NORM); i += 1
        v[i] = float(h.get("beam", 0)); i += 1
        v[i] = h.get("cos", 0.0); i += 1
        v[i] = h.get("sin", 0.0); i += 1
        v[i] = h.get("len", 0.0) / BEAM_LEN_NORM; i += 1
        v[i] = float(h.get("inside", 0)); i += 1
        v[i] = 1.0; i += 1
    i += (MAX_HAZARDS - len(obs["hazards"][:MAX_HAZARDS])) * HAZARD_DIM
    for s in obs.get("shots", [])[:MAX_SHOTS]:
        v[i] = s["dx"] / RANGE_NORM; i += 1
        v[i] = s["dy"] / RANGE_NORM; i += 1
        v[i] = s["vx"] / SHOT_SPEED_NORM; i += 1
        v[i] = s["vy"] / SHOT_SPEED_NORM; i += 1
        v[i] = 1.0; i += 1
    i += (MAX_SHOTS - len(obs.get("shots", [])[:MAX_SHOTS])) * SHOT_DIM
    for t in obs.get("traps", [])[:MAX_TRAPS]:
        v[i] = t["dx"] / RANGE_NORM; i += 1
        v[i] = t["dy"] / RANGE_NORM; i += 1
        v[i] = (t["kind"] + 1) / 3.0; i += 1   # 0=poison 1=sticky 2=explosive
        v[i] = 1.0; i += 1
    i += (MAX_TRAPS - len(obs.get("traps", [])[:MAX_TRAPS])) * TRAP_DIM
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
    v[base + N_RAYS + 1] = 1.0 if obs.get("aimBlocked") else 0.0   # ranged move, auto-target is behind cover
    i = base + N_RAYS + 2

    # own state: velocity, slow, invulnerability, altar power, temporary buffs, liquid underfoot
    s = obs.get("self") or {}
    v[i] = s.get("vx", 0.0) / PLAYER_VEL_NORM; i += 1
    v[i] = s.get("vy", 0.0) / PLAYER_VEL_NORM; i += 1
    v[i] = s.get("spd", 0.0) / PLAYER_VEL_NORM; i += 1
    v[i] = min(1.0, s.get("slow", 0.0) / 3.5); i += 1
    v[i] = min(1.0, s.get("invuln", 0.0) / 0.45); i += 1
    pw = s.get("power", -1)
    if 0 <= pw < POWERS:
        v[i + pw] = 1.0
    i += POWERS
    v[i] = s.get("powerLeft", 0.0); i += 1
    for b in (s.get("buffs") or [])[:BUFFS]:
        v[i] = min(1.0, b / 5.0); i += 1
    i += BUFFS - len((s.get("buffs") or [])[:BUFFS])
    v[i] = float(s.get("inLiquid", 0)); i += 1
    v[i] = s.get("liqSlow", 1.0); i += 1
    v[i] = float(s.get("liqDmg", 0)); i += 1
    v[i] = float(s.get("liqSlide", 0)); i += 1
    lq = s.get("liquid", -1)
    if 0 <= lq < len(LIQUID_KINDS):
        v[i + lq] = 1.0
    i += len(LIQUID_KINDS)

    # what we just did (the policy is feed-forward, so it cannot remember it otherwise)
    lm = int(obs.get("lastMove", 0))
    if 0 <= lm < N_MOVES:
        v[i + lm] = 1.0
    i += N_MOVES

    # springs (heal while standing in them), altars (timed power-up), big chests (stand next to them)
    for p in obs.get("specials", [])[:MAX_SPECIALS]:
        v[i] = p["dx"] / RANGE_NORM; i += 1
        v[i] = p["dy"] / RANGE_NORM; i += 1
        if p["type"] in SPECIAL_TYPES:
            v[i + SPECIAL_TYPES.index(p["type"])] = 1.0
        i += len(SPECIAL_TYPES)
        v[i] = p["state"]; i += 1                      # spring charge / chest opening progress / 1 for altar
        v[i] = (p["kind"] + 1) / 4.0 if p["type"] == "altar" else 0.0; i += 1
        v[i] = 1.0; i += 1
    i += (MAX_SPECIALS - len(obs.get("specials", [])[:MAX_SPECIALS])) * SPECIAL_DIM

    # distance to the nearest liquid in each of the 8 move directions
    for k, d in enumerate(obs.get("liqRays", [])[:N_RAYS]):
        v[i + k] = d / rmax
    i += N_RAYS

    w = obs.get("weather") or {}
    wk = w.get("kind", -1)
    if 0 <= wk < len(WEATHER_KINDS):
        v[i + wk] = 1.0
    i += len(WEATHER_KINDS)
    v[i] = w.get("level", 0.0); i += 1
    v[i] = float(w.get("hurts", 0)); i += 1

    d = obs.get("density") or {}
    for k, c in enumerate((d.get("sect") or [])[:8]):
        v[i + k] = min(1.0, c / 8.0)
    i += 8
    v[i] = min(2.0, d.get("total", 0) / 60.0); i += 1

    # time rift (portal to a legendary's arena) and the arena itself
    r = obs.get("rift") or {}
    v[i] = float(r.get("rift", 0)); i += 1
    v[i] = r.get("rdx", 0.0) / RANGE_NORM; i += 1
    v[i] = r.get("rdy", 0.0) / RANGE_NORM; i += 1
    v[i] = r.get("ttl", 0.0); i += 1
    v[i] = float(r.get("arena", 0)); i += 1
    v[i] = r.get("left", 0.0); i += 1
    v[i] = r.get("cdx", 0.0) / RANGE_NORM; i += 1
    v[i] = r.get("cdy", 0.0) / RANGE_NORM; i += 1
    v[i] = r.get("edge", 0.0); i += 1
    v[i] = float(r.get("won", 0)); i += 1
    assert i == OBS_DIM, (i, OBS_DIM)
    np.clip(v, -10.0, 10.0, out=v)
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
        if raw is not None:
            raw["lastMove"] = 0
        return _encode(raw), {}

    def step(self, action):
        move_idx, skill_idx, card_idx = int(action[0]), int(action[1]), int(action[2])
        raw = self._page.evaluate(
            "(a) => window.RL.step(a[0], a[1], a[2], a[3])",
            [move_idx, skill_idx, self.frame_skip, card_idx],
        )
        raw["lastMove"] = move_idx
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
        r -= max(0, cur.get("trapHits", 0) - prev.get("trapHits", 0)) * TRAP_PENALTY
        if cur.get("aimBlocked"):
            r -= AIM_BLOCKED_PENALTY
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
