"""Gymnasium environment wrapping the local PokeSurvivor JS game via Playwright.

v1 scope: fixed partner (Ditto, dex 132), level-up cards / items / bonds are
auto-picked by a heuristic inside the page (see game/js/rl_boot.js) so the
agent only controls movement + active-move switching.
"""
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from playwright.sync_api import sync_playwright

GAME_HTML = (Path(__file__).resolve().parent.parent / "game" / "rl_harness.html").as_uri()

MAX_ENEMIES = 8
MAX_HAZARDS = 6
RANGE_NORM = 600.0          # px used to squash dx/dy into roughly [-1, 1]
MAX_TIME = 30 * 60.0        # seconds; used only to normalize the time feature
OBS_DIM = 4 + 4 + 1 + MAX_ENEMIES * 5 + MAX_HAZARDS * 5

PARTNER_DEX = 132           # Ditto — fixed starter for v1 (see /memories/repo/plan.md)


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
    return v


class PokeSurvivorEnv(gym.Env):
    """One browser tab = one environment instance."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, headless: bool = True, speed: float = 10.0, frame_skip: int = 4,
                 max_time: float = MAX_TIME, partner_dex: int = PARTNER_DEX):
        super().__init__()
        self.headless = headless
        self.speed = speed
        self.frame_skip = frame_skip
        self.max_time = max_time
        self.partner_dex = partner_dex

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([9, 5])

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
        raw = self._page.evaluate("(dex) => window.RL.reset(dex)", self.partner_dex)
        self._page.evaluate("(k) => window.RL.setSpeed(k)", self.speed)
        self._prev_obs = raw
        return _encode(raw), {}

    def step(self, action):
        move_idx, skill_idx = int(action[0]), int(action[1])
        raw = self._page.evaluate(
            "(a) => window.RL.step(a[0], a[1], a[2])",
            [move_idx, skill_idx, self.frame_skip],
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
        r += max(0, cur["level"] - prev["level"]) * 5.0
        r += max(0, cur["kills"] - prev["kills"]) * 1.0
        hp_loss = max(0.0, prev["hp"] / max(1.0, prev["maxHp"]) - cur["hp"] / max(1.0, cur["maxHp"]))
        r -= hp_loss * 5.0
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
