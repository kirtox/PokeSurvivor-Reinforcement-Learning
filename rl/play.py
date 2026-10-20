"""Run a trained policy against the real (headed) browser window and show a
live telemetry dashboard (Window B) next to it: reward curve, HP/level/time,
and the current action (movement arrow + active-move highlight).

Usage:
  .venv\\Scripts\\python.exe -m rl.play --model rl/models/ppo_ditto_v1
"""
import argparse
import math
from collections import deque

import pygame
from stable_baselines3 import PPO

from rl.env import PokeSurvivorEnv

# Same order as DIRS in game/js/rl_boot.js: 0=stay, 1..8 = 8 compass dirs (screen coords, y down).
DIR_ANGLES = [None, -90, -45, 0, 45, 90, 135, 180, -135]

W, H = 420, 520
BG = (14, 18, 28)
FG = (230, 235, 245)
ACCENT = (90, 200, 255)
DANGER = (255, 90, 90)
OK = (120, 220, 140)


def draw_dashboard(screen, font, action, raw, reward_hist, last_pick=None):
    screen.fill(BG)
    move_idx, skill_idx = int(action[0]), int(action[1])

    # --- movement arrow ---
    cx, cy, r = 80, 100, 50
    pygame.draw.circle(screen, (30, 36, 50), (cx, cy), r, 2)
    angle = DIR_ANGLES[move_idx]
    if angle is None:
        pygame.draw.circle(screen, ACCENT, (cx, cy), 8)
    else:
        rad = math.radians(angle)
        tip = (cx + r * math.cos(rad), cy + r * math.sin(rad))
        left = (cx + 10 * math.cos(rad + 2.5), cy + 10 * math.sin(rad + 2.5))
        right = (cx + 10 * math.cos(rad - 2.5), cy + 10 * math.sin(rad - 2.5))
        pygame.draw.polygon(screen, ACCENT, [tip, left, right])
        pygame.draw.line(screen, ACCENT, (cx, cy), tip, 4)
    screen.blit(font.render("Move", True, FG), (cx - 22, cy + r + 10))

    # --- active move slot (1-4); skill_idx 0 = "no change" this step ---
    for slot in range(4):
        box = pygame.Rect(170 + slot * 55, 75, 45, 45)
        active = raw["active"] == slot
        just_switched = (skill_idx - 1) == slot
        color = ACCENT if active else (40, 46, 60)
        pygame.draw.rect(screen, color, box, border_radius=6)
        if just_switched:
            pygame.draw.rect(screen, OK, box, 3, border_radius=6)
        label = font.render(str(slot + 1), True, BG if active else FG)
        screen.blit(label, (box.centerx - 6, box.centery - 10))
    screen.blit(font.render("Active move", True, FG), (170, 125))

    # --- stats ---
    hp_frac = raw["hp"] / max(1.0, raw["maxHp"])
    stats = [
        f"Time:  {raw['time']:.1f}s",
        f"Level: {raw['level']}",
        f"Kills: {raw['kills']}",
        f"HP:    {raw['hp']:.0f}/{raw['maxHp']:.0f}",
        f"Card:  {last_pick or '-'}",
    ]
    for i, s in enumerate(stats):
        screen.blit(font.render(s, True, FG), (20, 170 + i * 26))
    bar = pygame.Rect(20, 295, 380, 18)
    pygame.draw.rect(screen, (40, 46, 60), bar, border_radius=4)
    fill = bar.copy(); fill.width = int(bar.width * max(0.0, hp_frac))
    pygame.draw.rect(screen, OK if hp_frac > 0.3 else DANGER, fill, border_radius=4)

    # --- reward curve ---
    screen.blit(font.render("Reward (recent steps)", True, FG), (20, 330))
    plot = pygame.Rect(20, 355, 380, 150)
    pygame.draw.rect(screen, (24, 28, 40), plot)
    if len(reward_hist) > 1:
        vals = list(reward_hist)
        lo, hi = min(vals), max(vals)
        span = max(1e-6, hi - lo)
        pts = []
        for i, v in enumerate(vals):
            x = plot.left + i / (len(vals) - 1) * plot.width
            y = plot.bottom - (v - lo) / span * plot.height
            pts.append((x, y))
        pygame.draw.lines(screen, ACCENT, False, pts, 2)
    pygame.display.flip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="rl/models/ppo_ditto_v1")
    args = ap.parse_args()

    model = PPO.load(args.model)
    env = PokeSurvivorEnv(headless=False, speed=1.0, frame_skip=4)  # Window A: real browser, real time

    pygame.init()
    screen = pygame.display.set_mode((W, H))                       # Window B: telemetry dashboard
    pygame.display.set_caption("PokeSurvivor RL — telemetry")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()
    reward_hist = deque(maxlen=200)
    last_pick = None

    obs, info = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running:
            break
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_hist.append(reward)
        if info["raw"].get("picked"):
            last_pick = info["raw"]["picked"]["title"]
        draw_dashboard(screen, font, action, info["raw"], reward_hist, last_pick)
        clock.tick(30)
        if terminated or truncated:
            obs, info = env.reset()
            reward_hist.clear()
            last_pick = None

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
