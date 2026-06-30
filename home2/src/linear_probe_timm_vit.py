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
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probe a pretrained ViT on CIFAR-10.")
    parser.add_argument("--dataset", choices=["CIFAR10"], default="CIFAR10")
    parser.add_argument("--model-name", default="vit_tiny_patch16_224")
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs" / "timm_linear_probe"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir, batch_size, num_workers):
    tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    for images, target in tqdm(loader, leave=False):
        images = images.to(device)
        output = model(images)
        features.append(output.cpu())
        labels.append(target)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


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

    train_loader, test_loader = build_loaders(args.data_dir, args.feature_batch_size, args.num_workers)
    backbone = timm.create_model(args.model_name, pretrained=True, num_classes=0).to(device)
    feature_dim = backbone.num_features

    train_features, train_labels = extract_features(backbone, train_loader, device)
    test_features, test_labels = extract_features(backbone, test_loader, device)

    train_feature_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_feature_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )

    classifier = nn.Linear(feature_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)

    history = []
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(classifier, train_feature_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(classifier, test_feature_loader, criterion, device)
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
            torch.save(classifier.state_dict(), output_dir / "best_model.pt")

    with (output_dir / "training_log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, output_dir)
    with (output_dir / "run_config.txt").open("w") as f:
        f.write(f"dataset={args.dataset}\n")
        f.write(f"model_name={args.model_name}\n")
        f.write("strategy=linear_probe\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"feature_dim={feature_dim}\n")


if __name__ == "__main__":
    main()
