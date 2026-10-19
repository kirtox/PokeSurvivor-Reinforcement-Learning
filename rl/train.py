"""PPO training entry point.

Usage:
  .venv\\Scripts\\python.exe -m rl.train --timesteps 1000000 --n-envs 8

Starts a small trial by default (1-2M steps) per the agreed success-criteria
plan: check the tensorboard learning curve before committing to a longer run.
"""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.env import PokeSurvivorEnv

MODELS_DIR = Path(__file__).resolve().parent / "models"
TB_DIR = Path(__file__).resolve().parent / "tb_logs"


def make_env():
    def _init():
        env = PokeSurvivorEnv(headless=True, speed=10.0, frame_skip=4)
        return Monitor(env)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--name", type=str, default="ppo_ditto_v1")
    ap.add_argument("--checkpoint-freq", type=int, default=50_000,
                    help="save a checkpoint every N total timesteps (0 to disable)")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=str(TB_DIR))
    callback = None
    if args.checkpoint_freq > 0:
        # save_freq counts calls per env, so divide by n_envs to get total timesteps
        callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(MODELS_DIR),
            name_prefix=args.name,
        )
    model.learn(total_timesteps=args.timesteps, tb_log_name=args.name, callback=callback)
    model.save(str(MODELS_DIR / args.name))
    vec_env.close()
    print(f"Saved model to {MODELS_DIR / args.name}.zip")
    print(f"Inspect learning curve with: tensorboard --logdir {TB_DIR}")


if __name__ == "__main__":
    main()
