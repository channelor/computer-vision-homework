import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"


def read_last_row(path: Path):
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def format_pct(value):
    return f"{float(value) * 100:.2f}%"


def main():
    experiments = {
        "from_scratch": OUTPUT_DIR / "from_scratch" / "training_log.csv",
        "finetune": OUTPUT_DIR / "finetune" / "training_log.csv",
    }

    lines = [
        "# Home1 Results Summary",
        "",
        "| Experiment | Final Epoch | Final Train Loss | Final Train Acc | Final Test Loss | Final Test Acc | Best Test Acc | Best Epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, log_path in experiments.items():
        if not log_path.exists():
            continue
        with log_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        row = rows[-1]
        best = max(rows, key=lambda item: float(item["test_acc"]))
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    row["epoch"],
                    f"{float(row['train_loss']):.4f}",
                    format_pct(row["train_acc"]),
                    f"{float(row['test_loss']):.4f}",
                    format_pct(row["test_acc"]),
                    format_pct(best["test_acc"]),
                    best["epoch"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "说明：本表只包含正式实验结果。训练集为 CIFAR-10 官方训练集，测试集为 CIFAR-10 官方测试集。",
            "",
        ]
    )
    output_path = OUTPUT_DIR / "results_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
