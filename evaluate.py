"""
evaluate.py
===========
Full evaluation script for the deepfake detection model.

Runs the trained model on a held-out test split and produces:
  1. Console output: Accuracy, Precision, Recall, F1-score, AUC-ROC (table format)
  2. outputs/metrics.txt        — metrics saved as text
  3. outputs/confusion_matrix.png — confusion matrix heatmap
  4. outputs/roc_curve.png      — ROC curve with AUC annotation
  5. outputs/gradcam_grid.png   — 6–8 sample images with predicted/true labels
                                   and Grad-CAM overlays (key screenshot for report)

Usage:
    python evaluate.py
    python evaluate.py --data_dir data --model checkpoints/model.pth --test_split 0.2
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image

from gradcam_utils import GradCAM, overlay_heatmap_on_image, pil_to_tensor


IMG_SIZE = 224


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained model from checkpoint. Must match train.py architecture."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'\n"
            "Run train.py first."
        )
    checkpoint    = torch.load(checkpoint_path, map_location=device)
    class_to_idx  = checkpoint.get("class_to_idx", {"fake": 0, "real": 1})

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_to_idx


@torch.no_grad()
def run_inference(model, loader, device, fake_idx: int):
    """
    Run the model over the test set.

    Returns:
        all_preds:  Predicted class indices (list of ints)
        all_labels: True class indices (list of ints)
        all_probs:  Predicted probability for the FAKE class (list of floats)
        all_tensors: Raw input tensors (for Grad-CAM grid)
        all_paths:  File paths of each sample
    """
    all_preds, all_labels, all_probs = [], [], []
    all_tensors, all_paths = [], []

    dataset = loader.dataset
    # Unwrap Subset to get the underlying dataset and indices
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = list(range(len(dataset)))

    for batch_imgs, batch_labels in loader:
        batch_imgs   = batch_imgs.to(device)
        logits       = model(batch_imgs)
        probs        = F.softmax(logits, dim=1)
        preds        = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.numpy())
        all_probs.extend(probs[:, fake_idx].cpu().numpy())
        all_tensors.extend(batch_imgs.cpu())

    # Collect file paths (for labelling the grad-cam grid)
    for idx in indices:
        path, _ = base_dataset.samples[idx]
        all_paths.append(path)

    return all_preds, all_labels, all_probs, all_tensors, all_paths


def print_metrics_table(acc, prec, rec, f1, auc):
    """Print metrics in a clean table — nice for terminal screenshots."""
    bar = "─" * 45
    print(f"\n  ╔{bar}╗")
    print(f"  ║  {'EVALUATION METRICS':^43}║")
    print(f"  ╠{bar}╣")
    print(f"  ║  {'Metric':<25} {'Value':>16}  ║")
    print(f"  ╠{bar}╣")
    print(f"  ║  {'Accuracy':<25} {acc*100:>15.2f}%  ║")
    print(f"  ║  {'Precision (FAKE)':<25} {prec*100:>15.2f}%  ║")
    print(f"  ║  {'Recall (FAKE)':<25} {rec*100:>15.2f}%  ║")
    print(f"  ║  {'F1-Score (FAKE)':<25} {f1*100:>15.2f}%  ║")
    print(f"  ║  {'AUC-ROC':<25} {auc:>16.4f}  ║")
    print(f"  ╚{bar}╝\n")


def save_confusion_matrix(y_true, y_pred, class_names: list, out_path: str):
    """Save a styled confusion matrix as a PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=12)

    # Print counts and normalized percentages in each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix — Deepfake Detection", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {out_path}")


def save_roc_curve(y_true, y_scores, auc: float, out_path: str):
    """Save the ROC curve with AUC annotated."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--',
            label="Random classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')

    # Mark the optimal threshold (closest to top-left corner)
    distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
    opt_idx   = np.argmin(distances)
    ax.scatter(fpr[opt_idx], tpr[opt_idx], color='red', zorder=5,
               label=f"Optimal threshold = {thresholds[opt_idx]:.3f}")

    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Deepfake Detection", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved       → {out_path}")


def save_gradcam_grid(
    model, tensors, paths, y_true, y_pred, fake_idx, device, out_path, n_samples=8
):
    """
    Create a grid of sample test images, each annotated with:
      - True label (green header if correct, red if wrong)
      - Predicted label + confidence
      - Grad-CAM heatmap overlay

    This is the KEY figure for the college report.
    """
    n_samples = min(n_samples, len(tensors))
    # Pick a balanced sample: equal real and fake where possible
    real_indices = [i for i, lbl in enumerate(y_true) if lbl != fake_idx]
    fake_indices = [i for i, lbl in enumerate(y_true) if lbl == fake_idx]

    n_each = n_samples // 2
    chosen = (real_indices[:n_each] + fake_indices[:n_each])[:n_samples]
    if len(chosen) < n_samples:
        # Pad with any remaining indices
        all_idx = list(range(len(tensors)))
        for i in all_idx:
            if i not in chosen and len(chosen) < n_samples:
                chosen.append(i)

    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 4))
    axes_flat = np.array(axes).flatten()

    target_layer = model.layer4[-1]

    for plot_i, sample_idx in enumerate(chosen):
        ax = axes_flat[plot_i]

        tensor     = tensors[sample_idx].unsqueeze(0).to(device)
        true_lbl   = y_true[sample_idx]
        pred_lbl   = y_pred[sample_idx]
        img_path   = paths[sample_idx]

        # Get confidence for the predicted class
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1)
        conf = probs[0, pred_lbl].item()

        # Grad-CAM for the PREDICTED class
        tensor_cam = tensor.clone().requires_grad_(True)
        gradcam = GradCAM(model=model, target_layer=target_layer)
        cam = gradcam.generate(tensor_cam, class_idx=pred_lbl)
        gradcam.remove_hooks()

        # Load and resize original image
        orig_img = Image.open(img_path).convert("RGB")
        orig_np  = np.array(orig_img.resize((IMG_SIZE, IMG_SIZE)))
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        overlay  = overlay_heatmap_on_image(orig_bgr, cam, alpha=0.45)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        ax.imshow(overlay_rgb)
        ax.axis("off")

        # Label text
        true_name = "REAL" if true_lbl != fake_idx else "FAKE"
        pred_name = "REAL" if pred_lbl != fake_idx else "FAKE"
        correct   = (true_lbl == pred_lbl)
        border_color = "#2ecc71" if correct else "#e74c3c"  # green / red

        title_text = f"True: {true_name}\nPred: {pred_name} ({conf*100:.1f}%)"
        ax.set_title(title_text, fontsize=9, fontweight='bold',
                     color=border_color, pad=4)

        # Draw coloured border around subplot
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

    # Hide any unused subplot axes
    for i in range(len(chosen), len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Correct prediction'),
        Patch(facecolor='#e74c3c', label='Wrong prediction'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Test Set Sample — Grad-CAM Explanations\n"
        "(Red/yellow regions = high model attention)",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grad-CAM grid saved   → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection model on test split"
    )
    parser.add_argument("--data_dir",    type=str, default="data",
                        help="Root folder with 'real/' and 'fake/' subfolders")
    parser.add_argument("--model",       type=str, default="checkpoints/model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--test_split",  type=float, default=0.2,
                        help="Fraction of data to use as test set (must match or be >= train.py val_split)")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out_dir",     type=str, default="outputs",
                        help="Directory for saved plots and metrics")
    parser.add_argument("--n_grid",      type=int, default=8,
                        help="Number of images in the Grad-CAM sample grid")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Deepfake Detection — Evaluation")
    print(f"{'='*55}")
    print(f"  Device    : {device}")
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Model     : {args.model}")
    print(f"  Test split: {args.test_split}")
    print(f"{'='*55}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, class_to_idx = load_model(args.model, device)
    fake_idx = class_to_idx.get("fake", 0)
    real_idx = class_to_idx.get("real", 1)
    class_names = ["FAKE", "REAL"] if fake_idx == 0 else ["REAL", "FAKE"]
    print(f"  Class mapping: {class_to_idx}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    full_dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=get_val_transforms(),
    )
    n_total = len(full_dataset)
    if n_total < 2:
        raise ValueError(f"Need at least 2 images in '{args.data_dir}'. Found {n_total}.")

    # Recreate the same test split used in train.py (same seed → same indices)
    indices = list(range(n_total))
    labels  = [full_dataset.targets[i] for i in indices]
    val_size = max(1, int(args.test_split * n_total))

    try:
        _, test_idx = train_test_split(
            indices, test_size=val_size,
            stratify=labels, random_state=args.seed
        )
    except ValueError:
        _, test_idx = train_test_split(
            indices, test_size=val_size, random_state=args.seed
        )

    test_subset = Subset(full_dataset, test_idx)
    bs = min(args.batch_size, len(test_subset))
    test_loader = DataLoader(test_subset, batch_size=bs, shuffle=False, num_workers=0)

    print(f"  Test samples: {len(test_idx)}\n")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("  Running inference on test set...")
    y_pred, y_true, y_scores, tensors, paths = run_inference(
        model, test_loader, device, fake_idx
    )
    print(f"  Done. {len(y_pred)} predictions collected.\n")

    # ── Compute metrics ───────────────────────────────────────────────────────
    acc   = accuracy_score(y_true, y_pred)

    # Use zero_division=0 to handle edge cases in tiny datasets
    prec  = precision_score(y_true, y_pred, pos_label=fake_idx, zero_division=0)
    rec   = recall_score(y_true, y_pred, pos_label=fake_idx, zero_division=0)
    f1    = f1_score(y_true, y_pred, pos_label=fake_idx, zero_division=0)

    # AUC-ROC requires at least one sample of each class
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = float('nan')
        print("  WARNING: AUC-ROC undefined (test set may contain only one class).")

    # ── Print metrics table ────────────────────────────────────────────────────
    print_metrics_table(acc, prec, rec, f1, auc)

    # Per-class breakdown
    print("  Per-class classification report:")
    print("  " + "-" * 45)
    target_names = [k.upper() for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
    report = classification_report(y_true, y_pred, target_names=target_names,
                                   zero_division=0)
    for line in report.split("\n"):
        print(f"    {line}")

    # ── Save metrics to text file ─────────────────────────────────────────────
    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Deepfake Detection — Evaluation Metrics\n")
        f.write("=" * 45 + "\n")
        f.write(f"Accuracy   : {acc*100:.2f}%\n")
        f.write(f"Precision  : {prec*100:.2f}%\n")
        f.write(f"Recall     : {rec*100:.2f}%\n")
        f.write(f"F1-Score   : {f1*100:.2f}%\n")
        f.write(f"AUC-ROC    : {auc:.4f}\n\n")
        f.write("Per-class report:\n")
        f.write(report)
    print(f"  Metrics saved         → {metrics_path}\n")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, class_names, cm_path)

    # ── ROC curve ─────────────────────────────────────────────────────────────
    if not np.isnan(auc):
        roc_path = os.path.join(args.out_dir, "roc_curve.png")
        save_roc_curve(y_true, y_scores, auc, roc_path)
    else:
        print("  Skipping ROC curve (AUC undefined).")

    # ── Grad-CAM sample grid ──────────────────────────────────────────────────
    grid_path = os.path.join(args.out_dir, "gradcam_grid.png")
    print(f"\n  Generating Grad-CAM grid ({args.n_grid} samples)...")
    save_gradcam_grid(
        model, tensors, paths, y_true, y_pred,
        fake_idx, device, grid_path, n_samples=args.n_grid
    )

    print(f"\n{'='*55}")
    print(f"  All outputs saved to: {args.out_dir}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
