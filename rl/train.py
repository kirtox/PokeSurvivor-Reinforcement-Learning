"""PPO training entry point.

Usage:
  .venv\\Scripts\\python.exe -m rl.train --timesteps 1000000 --n-envs 8
  .venv\\Scripts\\python.exe -m rl.train --timesteps 1500000 --name NAME --resume   # latest checkpoint
  .venv\\Scripts\\python.exe -m rl.train --timesteps 1500000 --name NAME --resume rl/models/NAME_249960_steps.zip

With --resume, --timesteps is the TOTAL target, not extra steps.

Starts a small trial by default (1-2M steps) per the agreed success-criteria
plan: check the tensorboard learning curve before committing to a longer run.
"""
import argparse
import json
import re
import time
from collections import deque
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.env import PokeSurvivorEnv

MODELS_DIR = Path(__file__).resolve().parent / "models"
TB_DIR = Path(__file__).resolve().parent / "tb_logs"


class StallWatchdog(BaseCallback):
    """Detects the known stall (env stepping several times slower and/or episodes
    collapsing to ~10 steps) and writes `flag_path`, then blocks so that rl.supervise can
    kill the process tree and resume from the last checkpoint. Only meant to run under
    rl.supervise (--watchdog-file). Placed before CheckpointCallback so a tripped step
    never saves a checkpoint.
    """
    WARMUP = 300      # vec steps ignored after start (browser start-up)
    WINDOW = 200      # vec steps in the speed window (~15 s when healthy)
    SLOW = 3.0        # trip if the window is this many times slower than the fastest seen
    EP_WINDOW = 24    # episodes in the length window
    EP_DROP = 0.25    # trip if mean episode length < this fraction of the best seen
    EP_MIN_BEST = 40  # ...only once episodes have been at least this long (steps)

    def __init__(self, flag_path):
        super().__init__()
        self.flag_path = Path(flag_path)
        # Baselines survive restarts: a run that starts already stalled must still be caught.
        self.base_path = self.flag_path.with_suffix(".baseline.json")
        self._last = None
        self._n = 0
        self._dts = deque(maxlen=self.WINDOW)
        self._best_dt = None
        self._eps = deque(maxlen=self.EP_WINDOW)
        self._best_ep = 0.0
        if self.base_path.exists():
            try:
                b = json.loads(self.base_path.read_text())
                self._best_dt = b.get("best_dt")
                self._best_ep = b.get("best_ep", 0.0)
            except (ValueError, OSError):
                pass

    def _save_baseline(self):
        self.base_path.write_text(json.dumps({"best_dt": self._best_dt, "best_ep": self._best_ep}))

    def _on_rollout_start(self):
        self._last = None  # don't count the PPO update as a slow env step

    def _on_step(self):
        now = time.monotonic()
        if self._last is not None:
            self._n += 1
            if self._n > self.WARMUP:
                self._dts.append(now - self._last)
        self._last = now
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._eps.append(ep["l"])

        if len(self._dts) == self.WINDOW:
            m = sum(self._dts) / self.WINDOW
            if self._best_dt is None or m < self._best_dt:
                self._best_dt = m
                self._save_baseline()
            if m > self.SLOW * self._best_dt:
                self._trip(f"env stepping {m / self._best_dt:.1f}x slower than best "
                           f"({m:.2f}s vs {self._best_dt:.2f}s per vec step) at {self.num_timesteps} steps")
        if len(self._eps) == self.EP_WINDOW:
            m = sum(self._eps) / self.EP_WINDOW
            if m > self._best_ep:
                self._best_ep = m
                self._save_baseline()
            if self._best_ep > self.EP_MIN_BEST and m < self.EP_DROP * self._best_ep:
                self._trip(f"episode length collapsed to {m:.1f} (best {self._best_ep:.1f}) "
                           f"at {self.num_timesteps} steps")
        return True

    def _trip(self, reason):
        msg = f"[watchdog] stall detected: {reason}"
        print(msg, flush=True)
        self.flag_path.write_text(msg)
        while True:  # wait for rl.supervise to kill this process
            time.sleep(60)


def make_env():
    def _init():
        env = PokeSurvivorEnv(headless=True, speed=10.0, frame_skip=4)
        return Monitor(env)
    return _init


def latest_checkpoint(name):
    best, best_steps = None, -1
    for p in MODELS_DIR.glob(f"{name}_*_steps.zip"):
        m = re.fullmatch(rf"{re.escape(name)}_(\d+)_steps\.zip", p.name)
        if m and int(m.group(1)) > best_steps:
            best, best_steps = p, int(m.group(1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--name", type=str, default="ppo_ditto_v1")
    ap.add_argument("--checkpoint-freq", type=int, default=50_000,
                    help="save a checkpoint every N total timesteps (0 to disable)")
    ap.add_argument("--resume", nargs="?", const="latest", default=None,
                    help="continue from a checkpoint .zip (no value = latest for --name); "
                         "--timesteps is then the total target")
    ap.add_argument("--watchdog-file", type=str, default=None,
                    help="enable the stall watchdog; it writes this file when tripped (used by rl.supervise)")
    args = ap.parse_args()

    # The MLP is tiny; extra torch threads only compete with the browser workers for CPU.
    torch.set_num_threads(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    if args.resume is None:
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=str(TB_DIR))
    else:
        ckpt = latest_checkpoint(args.name) if args.resume == "latest" else Path(args.resume)
        if ckpt is None or not ckpt.exists():
            vec_env.close()
            raise SystemExit(f"No checkpoint found to resume from ({args.resume!r}, name={args.name!r})")
        model = PPO.load(str(ckpt), env=vec_env, verbose=1, tensorboard_log=str(TB_DIR))
        print(f"Resumed from {ckpt} at {model.num_timesteps} timesteps")
    remaining = args.timesteps - model.num_timesteps
    if remaining <= 0:
        vec_env.close()
        raise SystemExit(f"Already at {model.num_timesteps} >= target {args.timesteps}; nothing to do")
    callbacks = []
    if args.watchdog_file:
        callbacks.append(StallWatchdog(args.watchdog_file))
    if args.checkpoint_freq > 0:
        # save_freq counts calls per env, so divide by n_envs to get total timesteps
        callbacks.append(CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(MODELS_DIR),
            name_prefix=args.name,
        ))
    callback = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=remaining, tb_log_name=args.name, callback=callback,
                reset_num_timesteps=args.resume is None)
    model.save(str(MODELS_DIR / args.name))
    vec_env.close()
    print(f"Saved model to {MODELS_DIR / args.name}.zip")
    print(f"Inspect learning curve with: tensorboard --logdir {TB_DIR}")


if __name__ == "__main__":
    main()
