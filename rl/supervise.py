"""Runs rl.train and restarts it from the latest checkpoint when the stall watchdog trips.

Usage (same arguments as rl.train):
  .venv\\Scripts\\python.exe -m rl.supervise --name mewtwo_v2 --timesteps 500000 --n-envs 12 --checkpoint-freq 25000
  .venv\\Scripts\\python.exe -m rl.supervise --name mewtwo_v2 --timesteps 500000 --n-envs 12 --checkpoint-freq 25000 --resume rl/models/mewtwo_v2_224964_steps.zip

Extra option: --max-restarts N (default 10).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from rl.train import MODELS_DIR, latest_checkpoint


def kill_tree(proc):
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--resume", nargs="?", const="latest", default=None)
    ap.add_argument("--max-restarts", type=int, default=10)
    ap.add_argument("--restart-delay", type=int, default=300,
                    help="seconds to wait after a stall before restarting (lets a transient host slowdown pass)")
    args, passthrough = ap.parse_known_args()

    flag = MODELS_DIR.parent / f".stall_{args.name}"
    base = [sys.executable, "-m", "rl.train", "--name", args.name, *passthrough,
            "--watchdog-file", str(flag)]
    resume = args.resume
    if resume == "latest":
        ckpt = latest_checkpoint(args.name)
        resume = str(ckpt) if ckpt else None

    for attempt in range(args.max_restarts + 1):
        flag.unlink(missing_ok=True)
        cmd = base + (["--resume", resume] if resume else [])
        print(f"[supervise] start #{attempt}: {' '.join(cmd[3:])}", flush=True)
        proc = subprocess.Popen(cmd)
        while proc.poll() is None and not flag.exists():
            time.sleep(5)
        if not flag.exists():
            sys.exit(proc.returncode)
        print(f"[supervise] {flag.read_text()}", flush=True)
        kill_tree(proc)
        flag.unlink(missing_ok=True)
        print(f"[supervise] waiting {args.restart_delay}s before restart", flush=True)
        time.sleep(args.restart_delay)
        ckpt = latest_checkpoint(args.name)
        if ckpt is None:
            sys.exit("[supervise] no checkpoint to resume from")
        resume = str(ckpt)
        print(f"[supervise] restarting from {resume}", flush=True)
    sys.exit(f"[supervise] gave up after {args.max_restarts} restarts")


if __name__ == "__main__":
    main()
