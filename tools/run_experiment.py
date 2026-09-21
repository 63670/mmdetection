"""Train an MMDetection experiment and evaluate its best checkpoint on the test split."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    """Parse single-GPU experiment arguments."""
    parser = argparse.ArgumentParser(
        description="Train an MMDetection config, then test its best checkpoint."
    )
    parser.add_argument("config", help="Configuration path relative to the repository root.")
    parser.add_argument("--work-dir", required=True, help="Directory for both training and test outputs.")
    parser.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value for this single-process run.")
    parser.add_argument("--seed", type=int, help="Random seed passed to MMEngine as randomness.seed.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP during training.")
    parser.add_argument("--auto-scale-lr", action="store_true", help="Enable automatic LR scaling during training.")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        help="Optional config overrides forwarded to tools/train.py.",
    )
    return parser.parse_args()


def run(command, env, log_path=None):
    """Run a command, optionally teeing its combined output to a log file."""
    print("+", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        return

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def find_best_checkpoint(work_dir):
    """Return the newest checkpoint selected by the COCO bbox mAP hook."""
    checkpoints = list(work_dir.glob("best_coco_bbox_mAP_*.pth"))
    if not checkpoints:
        checkpoints = list(work_dir.glob("best_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No best checkpoint was found in {work_dir}. "
            "Check that the config defines CheckpointHook.save_best."
        )
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def remove_extra_checkpoints(work_dir, best_checkpoint):
    """Remove saved model checkpoints after successful testing, retaining only the best one."""
    deleted = []
    for checkpoint in work_dir.glob("*.pth"):
        if checkpoint.resolve() != best_checkpoint.resolve():
            checkpoint.unlink()
            deleted.append(checkpoint.name)
    last_checkpoint = work_dir / "last_checkpoint"
    if last_checkpoint.exists():
        last_checkpoint.unlink()
        deleted.append(last_checkpoint.name)
    (work_dir / "deleted_checkpoints.log").write_text(
        "\n".join(deleted) + ("\n" if deleted else ""), encoding="utf-8"
    )


def main():
    """Train and test one experiment without separating its output directories."""
    args = parse_args()
    config = Path(args.config)
    if not config.is_absolute():
        config = REPO_ROOT / config
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")

    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = REPO_ROOT / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device

    train_command = [
        sys.executable,
        "tools/train.py",
        str(config),
        "--work-dir",
        str(work_dir),
    ]
    if args.amp:
        train_command.append("--amp")
    if args.auto_scale_lr:
        train_command.append("--auto-scale-lr")
    cfg_options = list(args.cfg_options or [])
    if args.seed is not None:
        cfg_options.append(f"randomness.seed={args.seed}")
        (work_dir / "seed.txt").write_text(f"{args.seed}\n", encoding="utf-8")
    if cfg_options:
        train_command.extend(["--cfg-options", *cfg_options])
    run(train_command, env)

    checkpoint = find_best_checkpoint(work_dir)
    (work_dir / "test_checkpoint.txt").write_text(f"{checkpoint}\n", encoding="utf-8")
    test_command = [
        sys.executable,
        "tools/test.py",
        str(config),
        str(checkpoint),
        "--work-dir",
        str(work_dir),
    ]
    run(test_command, env, work_dir / "test_metrics.log")
    remove_extra_checkpoints(work_dir, checkpoint)


if __name__ == "__main__":
    main()
