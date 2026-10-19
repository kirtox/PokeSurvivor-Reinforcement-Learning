"""Quick manual check: does the local game load headless and respond to RL.step()?
Run with: .venv\\Scripts\\python.exe rl\\smoke_test.py
"""
import time
from rl.env import PokeSurvivorEnv

env = PokeSurvivorEnv(headless=True, speed=10.0, frame_skip=4)
obs, info = env.reset()
print("reset ok, obs shape:", obs.shape)

t0 = time.time()
total_r = 0.0
for i in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_r += reward
    if i % 50 == 0:
        raw = info["raw"]
        print(f"step {i}: sim_time={raw['time']:.1f}s hp={raw['hp']:.0f}/{raw['maxHp']:.0f} "
              f"level={raw['level']} dead={raw['dead']}")
    if terminated or truncated:
        print("episode ended at step", i)
        break

print(f"200 steps in {time.time() - t0:.2f}s real time, total reward={total_r:.2f}")
env.close()
