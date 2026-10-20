import sys
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from rl.env import PokeSurvivorEnv

EPISODES = 10
MAX_STEPS = 2500


def run(name, policy, episodes=EPISODES):
    env = PokeSurvivorEnv(headless=True, speed=10.0, frame_skip=4)
    rows, picks, slots_kinds = [], Counter(), Counter()
    for ep in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(MAX_STEPS):
            obs, r, term, trunc, info = env.step(policy(env, obs))
            total += r
            raw = info["raw"]
            if raw.get("picked"):
                picks[raw["picked"]["type"]] += 1
            if term or trunc:
                break
        for m in raw["moves"]:
            slots_kinds[m["kind"]] += 1
        rows.append((raw["time"], raw["level"], raw["kills"], total))
    env.close()
    a = np.array(rows)
    print(f"[{name}] survival={a[:,0].mean():.1f}s (+/-{a[:,0].std():.1f}) level={a[:,1].mean():.1f} "
          f"kills={a[:,2].mean():.1f} reward={a[:,3].mean():.2f}")
    print(f"[{name}] card types picked: {dict(picks)}  final moves: {dict(slots_kinds)}")


if __name__ == "__main__":
    model = PPO.load(sys.argv[1])
    run("model", lambda env, obs: model.predict(obs, deterministic=True)[0])
    run("random", lambda env, obs: env.action_space.sample())
    run("still", lambda env, obs: np.array([0, np.random.randint(0, 5), 0]))
