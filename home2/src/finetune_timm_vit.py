import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained ViT from timm.")
    parser.add_argument("--dataset", choices=["CIFAR10"], default="CIFAR10")
    parser.add_argument("--model-name", default="vit_tiny_patch16_224")
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs" / "timm_finetune"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--finetune-strategy",
        choices=["full", "head_only"],
        default="full",
        help="full: update all parameters; head_only: freeze backbone and train classifier only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir, batch_size, num_workers):
    train_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def configure_trainable_parameters(model, strategy):
    if strategy == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif strategy == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            for param in classifier.parameters():
                param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                if name.startswith("head."):
                    param.requires_grad = True
    else:
        raise ValueError(f"Unknown finetune strategy: {strategy}")

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def save_curves(history, output_dir):
    epochs = [row["epoch"] for row in history]
    for metric in ["loss", "acc"]:
        plt.figure()
        plt.plot(epochs, [row[f"train_{metric}"] for row in history], label=f"train_{metric}")
        plt.plot(epochs, [row[f"test_{metric}"] for row in history], label=f"test_{metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_curve.png")
        plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    train_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = timm.create_model(args.model_name, pretrained=True, num_classes=10).to(device)
    trainable_params, total_params = configure_trainable_parameters(model, args.finetune_strategy)
    print(
        {
            "model": args.model_name,
            "strategy": args.finetune_strategy,
            "trainable_params": trainable_params,
            "total_params": total_params,
        }
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.05,
    )

    history = []
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)
        print(row)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    with (output_dir / "training_log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, output_dir)
    with (output_dir / "run_config.txt").open("w") as f:
        f.write(f"dataset={args.dataset}\n")
        f.write(f"model_name={args.model_name}\n")
        f.write(f"finetune_strategy={args.finetune_strategy}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"trainable_params={trainable_params}\n")
        f.write(f"total_params={total_params}\n")


if __name__ == "__main__":
    main()
