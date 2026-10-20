"""Baselines to compare the trained agent against: run_baselines.py

Usage: .venv\\Scripts\\python.exe -m rl.baseline
"""
import numpy as np
from rl.env import PokeSurvivorEnv

EPISODES = 10
MAX_STEPS = 2000  # safety cap per episode


def run_policy(name, pick_action, episodes=EPISODES):
    env = PokeSurvivorEnv(headless=True, speed=10.0, frame_skip=4)
    survival_times, rewards = [], []
    for ep in range(episodes):
        obs, info = env.reset()
        total_r = 0.0
        for _ in range(MAX_STEPS):
            action = pick_action(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            if terminated or truncated:
                break
        survival_times.append(info["raw"]["time"])
        rewards.append(total_r)
        print(f"[{name}] episode {ep}: survived {info['raw']['time']:.1f}s, reward={total_r:.2f}")
    env.close()
    print(f"[{name}] mean survival={np.mean(survival_times):.1f}s "
          f"(+/-{np.std(survival_times):.1f}), mean reward={np.mean(rewards):.2f}")
    return survival_times, rewards


def random_action(env, obs):
    return env.action_space.sample()


def stand_still_action(env, obs):
    # move_idx=0 -> stay put; still cycles the active move so it isn't fully inert; card 0 = first offered.
    return np.array([0, np.random.randint(0, 5), 0])


if __name__ == "__main__":
    run_policy("random", random_action)
    run_policy("stand_still", stand_still_action)
