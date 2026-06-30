from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run ViT position encoding ablation experiments.")
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-root", default=str(PROJECT_DIR / "outputs" / "position_ablation"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("with_pos_embed", []),
        ("without_pos_embed", ["--no-pos-embed"]),
    ]
    for name, extra_args in experiments:
        cmd = [
            sys.executable,
            "train_vit.py",
            "--dataset",
            args.dataset,
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
            "--output-dir",
            str(output_root / name),
            *extra_args,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


if __name__ == "__main__":
    main()
