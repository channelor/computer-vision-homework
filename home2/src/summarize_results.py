import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def format_pct(value):
    return f"{float(value) * 100:.2f}%"


def summarize_experiment(name, path):
    rows = read_rows(path)
    if not rows:
        return None
    final = rows[-1]
    best = max(rows, key=lambda row: float(row["test_acc"]))
    return {
        "name": name,
        "final_epoch": final["epoch"],
        "final_train_loss": final["train_loss"],
        "final_train_acc": final["train_acc"],
        "final_test_loss": final["test_loss"],
        "final_test_acc": final["test_acc"],
        "best_test_acc": best["test_acc"],
        "best_epoch": best["epoch"],
    }


def main():
    experiments = {
        "vit_from_scratch": OUTPUT_DIR / "vit_from_scratch" / "training_log.csv",
        "timm_finetune_full": OUTPUT_DIR / "timm_finetune" / "training_log.csv",
        "timm_finetune_head_only": OUTPUT_DIR / "timm_finetune_head_only" / "training_log.csv",
        "timm_linear_probe": OUTPUT_DIR / "timm_linear_probe" / "training_log.csv",
        "position_with_pos_embed": OUTPUT_DIR / "position_ablation" / "with_pos_embed" / "training_log.csv",
        "position_without_pos_embed": OUTPUT_DIR / "position_ablation" / "without_pos_embed" / "training_log.csv",
    }

    lines = [
        "# Home2 Results Summary",
        "",
        "| Experiment | Final Epoch | Final Train Loss | Final Train Acc | Final Test Loss | Final Test Acc | Best Test Acc | Best Epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, path in experiments.items():
        result = summarize_experiment(name, path)
        if result is None:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    result["name"],
                    result["final_epoch"],
                    f"{float(result['final_train_loss']):.4f}",
                    format_pct(result["final_train_acc"]),
                    f"{float(result['final_test_loss']):.4f}",
                    format_pct(result["final_test_acc"]),
                    format_pct(result["best_test_acc"]),
                    result["best_epoch"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "说明：本表汇总 ViT 从零训练、预训练 ViT 微调策略对比和位置编码消融实验结果。",
            "",
        ]
    )
    output_path = OUTPUT_DIR / "results_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
