import argparse
import csv
from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a small photo of a {}.",
    "a low resolution photo of a {}.",
    "a cropped photo of a {}.",
]

PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP zero-shot classification on CIFAR-10.")
    parser.add_argument("--dataset", choices=["CIFAR10"], default="CIFAR10")
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs" / "clip_zero_shot"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-examples", type=int, default=60)
    parser.add_argument("--single-prompt", action="store_true", help="Use only one prompt per class.")
    return parser.parse_args()


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_loader(data_dir, batch_size, num_workers, preprocess):
    dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, len(dataset)


@torch.no_grad()
def build_text_features(model, tokenizer, classes, device, single_prompt=False):
    templates = PROMPT_TEMPLATES[:1] if single_prompt else PROMPT_TEMPLATES
    class_features = []
    prompts_by_class = {}

    for class_name in classes:
        prompts = [template.format(class_name) for template in templates]
        prompts_by_class[class_name] = prompts
        text = tokenizer(prompts).to(device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_feature = text_features.mean(dim=0)
        text_feature = text_feature / text_feature.norm()
        class_features.append(text_feature)

    return torch.stack(class_features, dim=0), prompts_by_class


@torch.no_grad()
def evaluate(model, loader, text_features, classes, device, max_examples):
    correct = 0
    total = 0
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.long)
    examples = []

    for images, labels in tqdm(loader):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=1).cpu()
        preds = probs.argmax(dim=1)

        for label, pred, prob_row in zip(labels, preds, probs):
            label_idx = int(label)
            pred_idx = int(pred)
            confusion[label_idx, pred_idx] += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "label": classes[label_idx],
                        "prediction": classes[pred_idx],
                        "correct": label_idx == pred_idx,
                        "confidence": float(prob_row[pred_idx]),
                    }
                )

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total, confusion, examples


def write_confusion_matrix(path, classes, confusion):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label/pred", *classes])
        for class_name, row in zip(classes, confusion.tolist()):
            writer.writerow([class_name, *row])


def write_per_class_accuracy(path, classes, confusion):
    rows = []
    for idx, class_name in enumerate(classes):
        total = int(confusion[idx].sum().item())
        correct = int(confusion[idx, idx].item())
        accuracy = correct / total if total else 0.0
        rows.append({"class": class_name, "correct": correct, "total": total, "accuracy": accuracy})
    rows.sort(key=lambda row: row["accuracy"], reverse=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "correct", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_top_confusions(path, classes, confusion):
    rows = []
    for label_idx, label_name in enumerate(classes):
        for pred_idx, pred_name in enumerate(classes):
            if label_idx == pred_idx:
                continue
            count = int(confusion[label_idx, pred_idx].item())
            if count > 0:
                rows.append({"label": label_name, "prediction": pred_name, "count": count})
    rows.sort(key=lambda row: row["count"], reverse=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "prediction", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_examples(path, examples):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "prediction", "correct", "confidence"])
        writer.writeheader()
        writer.writerows(examples)


def write_prompts(path, prompts_by_class):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "prompt"])
        for class_name, prompts in prompts_by_class.items():
            for prompt in prompts:
                writer.writerow([class_name, prompt])


def write_summary(path, args, dataset_size, accuracy, per_class_rows, confusion_rows):
    best_classes = per_class_rows[:3]
    worst_classes = per_class_rows[-3:]
    top_confusions = confusion_rows[:5]

    lines = [
        "# CLIP Zero-Shot Classification Summary",
        "",
        f"- Dataset: {args.dataset}",
        f"- Test samples: {dataset_size}",
        f"- Model: {args.model_name}",
        f"- Pretrained weights: {args.pretrained}",
        f"- Prompt mode: {'single prompt' if args.single_prompt else 'prompt ensemble'}",
        f"- Accuracy: {accuracy:.4f}",
        "",
        "## Best Classes",
        "",
    ]
    for row in best_classes:
        lines.append(f"- {row['class']}: {row['accuracy']:.4f} ({row['correct']}/{row['total']})")

    lines.extend(["", "## Worst Classes", ""])
    for row in worst_classes:
        lines.append(f"- {row['class']}: {row['accuracy']:.4f} ({row['correct']}/{row['total']})")

    lines.extend(["", "## Top Confusions", ""])
    for row in top_confusions:
        lines.append(f"- {row['label']} -> {row['prediction']}: {row['count']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device).eval()

    loader, dataset_size = build_loader(args.data_dir, args.batch_size, args.num_workers, preprocess)
    text_features, prompts_by_class = build_text_features(
        model,
        tokenizer,
        CIFAR10_CLASSES,
        device,
        single_prompt=args.single_prompt,
    )
    accuracy, confusion, examples = evaluate(
        model,
        loader,
        text_features,
        CIFAR10_CLASSES,
        device,
        args.max_examples,
    )

    write_confusion_matrix(output_dir / "confusion_matrix.csv", CIFAR10_CLASSES, confusion)
    per_class_rows = write_per_class_accuracy(output_dir / "per_class_accuracy.csv", CIFAR10_CLASSES, confusion)
    confusion_rows = write_top_confusions(output_dir / "top_confusions.csv", CIFAR10_CLASSES, confusion)
    write_examples(output_dir / "examples.csv", examples)
    write_prompts(output_dir / "prompts.csv", prompts_by_class)
    write_summary(output_dir / "summary.md", args, dataset_size, accuracy, per_class_rows, confusion_rows)

    print(f"Zero-shot accuracy: {accuracy:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
