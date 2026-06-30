import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run full and head-only ViT fine-tuning strategies.")
    parser.add_argument("--dataset", choices=["CIFAR10"], default="CIFAR10")
    parser.add_argument("--model-name", default="vit_tiny_patch16_224")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_strategy(args, strategy, output_dir):
    log_path = output_dir / "training_log.csv"
    if args.skip_existing and log_path.exists():
        print(f"Skip existing result: {log_path}")
        return
    cmd = [
        sys.executable,
        "finetune_timm_vit.py",
        "--dataset",
        args.dataset,
        "--model-name",
        args.model_name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--finetune-strategy",
        strategy,
        "--output-dir",
        str(output_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def main():
    args = parse_args()
    run_strategy(args, "full", PROJECT_DIR / "outputs" / "timm_finetune")
    run_strategy(args, "head_only", PROJECT_DIR / "outputs" / "timm_finetune_head_only")


if __name__ == "__main__":
    main()
