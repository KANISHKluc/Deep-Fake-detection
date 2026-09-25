"""
train.py
========
Fine-tune a pretrained ResNet-18 binary classifier (REAL=0, FAKE=1)
on a small dataset of real and AI-generated face images.

Only the final classification head is trained; the backbone is frozen.
This keeps training time under ~5 minutes on CPU for tiny datasets.

Usage:
    python train.py
    python train.py --data_dir data --epochs 15 --batch_size 8 --lr 0.001

Expected data layout:
    data/
        real/   ← any .jpg/.jpeg/.png real face images
        fake/   ← any .jpg/.jpeg/.png fake/AI-generated face images
"""

import os
import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE   = 224   # ResNet expects 224×224
NUM_CLASSES = 2    # REAL (0) or FAKE (1)


def get_transforms(train: bool) -> transforms.Compose:
    """
    Return data augmentation transforms for training,
    or simple resize+normalize for validation/test.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


def build_model(device: torch.device) -> nn.Module:
    """
    Load a pretrained ResNet-18 and replace its final fully-connected layer
    with a new binary classification head.

    The backbone weights (all layers except fc) are FROZEN — only the new
    head is updated during training. This is called "feature extraction"
    and is ideal for small datasets because it prevents overfitting.
    """
    # Load ResNet-18 with ImageNet-pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze ALL backbone parameters so their gradients are not computed
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final FC layer (originally 512 → 1000 for ImageNet)
    # with a new head: 512 → 2 (binary: REAL vs FAKE)
    in_features = model.fc.in_features   # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),               # regularisation for small datasets
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, NUM_CLASSES),     # logits for 2 classes
    )
    # New head parameters have requires_grad=True by default

    return model.to(device)


def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance.
    Each class gets equal expected sampling frequency, so even if we have
    12 real and 8 fake images, batches won't be dominated by one class.
    """
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(targets)
    # Weight each sample inversely proportional to its class frequency
    weights = 1.0 / class_counts[targets]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)          # (batch, 2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc  = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc  = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(
        description="Train deepfake detection model (frozen ResNet-18 + binary head)"
    )
    parser.add_argument("--data_dir",   type=str, default="data",
                        help="Root folder with 'real/' and 'fake/' subfolders")
    parser.add_argument("--epochs",     type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (keep small for tiny datasets)")
    parser.add_argument("--lr",         type=float, default=5e-4,
                        help="Learning rate for the classification head")
    parser.add_argument("--val_split",  type=float, default=0.2,
                        help="Fraction of data to use for validation")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_path",  type=str, default="checkpoints/model.pth",
                        help="Where to save the best model weights")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Deepfake Detection — Training")
    print(f"{'='*55}")
    print(f"  Device    : {device}")
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR        : {args.lr}")
    print(f"{'='*55}\n")

    # ── Load full dataset (using train transforms; we override later for val) ──
    full_dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=get_transforms(train=True),
    )

    # Validate expected class mapping: ImageFolder sorts folders alphabetically
    # So: fake=0, real=1  (alphabetical). We remap below if needed.
    print(f"  Class mapping (folder → index): {full_dataset.class_to_idx}")
    # Expected: {'fake': 0, 'real': 1}
    # REAL=0, FAKE=1 convention is used in inference; mapping is handled via
    # the saved class_to_idx in the checkpoint.

    n_total = len(full_dataset)
    if n_total < 4:
        raise ValueError(
            f"Only {n_total} images found in '{args.data_dir}'.\n"
            "Please add images to data/real/ and data/fake/ first.\n"
            "See README.md for instructions."
        )

    print(f"  Total images : {n_total}")
    for cls, idx in full_dataset.class_to_idx.items():
        count = sum(1 for _, lbl in full_dataset.samples if lbl == idx)
        print(f"    {cls:>6}: {count} images  (label={idx})")

    # ── Train / Validation split (stratified) ─────────────────────────────────
    indices = list(range(n_total))
    labels  = [full_dataset.targets[i] for i in indices]

    # Clamp val_split so we always have at least 1 sample per split
    val_size = max(1, int(args.val_split * n_total))
    train_size = n_total - val_size

    if train_size < 1:
        # Dataset too tiny — use all for training, val on same data (demo only)
        train_idx, val_idx = indices, indices
        print("\n  WARNING: Dataset too small for a proper split. "
              "Using all data for both train and val.")
    else:
        try:
            train_idx, val_idx = train_test_split(
                indices, test_size=val_size,
                stratify=labels, random_state=args.seed
            )
        except ValueError:
            # stratify fails if a class has only 1 sample
            train_idx, val_idx = train_test_split(
                indices, test_size=val_size, random_state=args.seed
            )

    print(f"\n  Train samples: {len(train_idx)}  |  Val samples: {len(val_idx)}\n")

    # Apply different transforms to train vs val subsets
    train_dataset = copy.copy(full_dataset)
    train_dataset.transform = get_transforms(train=True)
    val_dataset = copy.copy(full_dataset)
    val_dataset.transform = get_transforms(train=False)

    train_subset = Subset(train_dataset, train_idx)
    val_subset   = Subset(val_dataset, val_idx)

    # Use weighted sampler on train to handle class imbalance
    train_sampler = make_weighted_sampler(
        Subset(full_dataset, train_idx)
    )

    # batch_size must not exceed dataset size
    bs = min(args.batch_size, len(train_subset))

    train_loader = DataLoader(
        train_subset,
        batch_size=bs,
        sampler=train_sampler,
        num_workers=0,   # 0 is safest on Windows
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss()

    # Only train the new classification head (backbone is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
    # Reduce LR if validation loss plateaus for 5 consecutive epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}")
    print(f"  {'-'*50}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        # Save best model
        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())

        marker = " ★" if v_acc >= best_val_acc else ""
        print(f"  {epoch:>5}  {t_loss:>10.4f}  {t_acc:>9.4f}  "
              f"{v_loss:>9.4f}  {v_acc:>8.4f}{marker}")

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s  |  Best val acc: {best_val_acc:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.load_state_dict(best_weights)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx":     full_dataset.class_to_idx,
        "img_size":         IMG_SIZE,
        "history":          history,
    }, args.save_path)
    print(f"  Model saved → {args.save_path}\n")

    # ── Plot training curves ──────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        os.makedirs("outputs", exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, args.epochs + 1)

        axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker='o')
        axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",   marker='s')
        axes[0].set_title("Loss per Epoch")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_range, history["train_acc"], label="Train Acc", marker='o')
        axes[1].plot(epochs_range, history["val_acc"],   label="Val Acc",   marker='s')
        axes[1].set_title("Accuracy per Epoch")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.suptitle("Training History — Deepfake Detection", fontsize=13, y=1.02)
        plt.tight_layout()
        curve_path = "outputs/training_curves.png"
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Training curves saved → {curve_path}\n")
    except Exception as e:
        print(f"  (Could not save training curves: {e})\n")


if __name__ == "__main__":
    main()
