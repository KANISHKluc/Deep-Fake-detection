"""
predict.py
==========
Single-image inference script.

Given a path to a face image, this script:
  1. Loads the trained model from checkpoints/model.pth
  2. Runs the image through the model to get a prediction
  3. Generates a Grad-CAM heatmap showing WHICH regions influenced the decision
  4. Saves a side-by-side comparison image: original | heatmap overlay
  5. Prints the predicted label (REAL / FAKE) and confidence score

Usage:
    python predict.py --image path/to/face.jpg
    python predict.py --image path/to/face.jpg --model checkpoints/model.pth --out outputs/result.png
"""

import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import models
import torch.nn as nn

from gradcam_utils import GradCAM, overlay_heatmap_on_image, pil_to_tensor


def load_model(checkpoint_path: str, device: torch.device):
    """
    Reconstruct the ResNet-18 + binary head architecture and load saved weights.

    Returns:
        model:        The loaded model in eval mode.
        class_to_idx: Dict mapping class name → label index (e.g. {'fake':0,'real':1}).
        img_size:     Input image size expected by the model.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'\n"
            "Run train.py first to create a trained model."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {"fake": 0, "real": 1})
    img_size     = checkpoint.get("img_size", 224)

    # Rebuild identical architecture (must match train.py)
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

    return model, class_to_idx, img_size


def predict_image(model, tensor: torch.Tensor, class_to_idx: dict, device: torch.device):
    """
    Run inference and return prediction details.

    Returns:
        pred_label:  'REAL' or 'FAKE'
        confidence:  Probability of the predicted class (0–1)
        fake_prob:   Probability of FAKE class specifically (used for ROC AUC)
        pred_idx:    Raw predicted class index
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)                # (1, 2)
        probs  = F.softmax(logits, dim=1)     # convert logits → probabilities

    # Determine which index corresponds to 'fake' and 'real'
    fake_idx = class_to_idx.get("fake", 0)
    real_idx = class_to_idx.get("real", 1)

    pred_idx   = probs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()
    fake_prob  = probs[0, fake_idx].item()

    pred_label = "FAKE" if pred_idx == fake_idx else "REAL"
    return pred_label, confidence, fake_prob, pred_idx


def main():
    parser = argparse.ArgumentParser(
        description="Deepfake detection: single-image inference with Grad-CAM"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input face image")
    parser.add_argument("--model", type=str, default="checkpoints/model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--out",   type=str, default=None,
                        help="Output path for the result image (auto-generated if omitted)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Deepfake Detection — Single Image Inference")
    print(f"{'='*55}")
    print(f"  Device    : {device}")
    print(f"  Image     : {args.image}")
    print(f"  Model     : {args.model}")

    model, class_to_idx, img_size = load_model(args.model, device)
    print(f"  Classes   : {class_to_idx}")

    # ── Load and preprocess image ─────────────────────────────────────────────
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: '{args.image}'")

    pil_img = Image.open(args.image).convert("RGB")
    tensor  = pil_to_tensor(pil_img, img_size=img_size).to(device)
    tensor.requires_grad_(True)   # needed for Grad-CAM backward pass

    # ── Predict ───────────────────────────────────────────────────────────────
    pred_label, confidence, fake_prob, pred_idx = predict_image(
        model, tensor, class_to_idx, device
    )

    print(f"\n{'─'*55}")
    print(f"  Prediction : {pred_label}")
    print(f"  Confidence : {confidence*100:.2f}%")
    print(f"  FAKE prob  : {fake_prob*100:.2f}%")
    print(f"  REAL prob  : {(1-fake_prob)*100:.2f}%")
    print(f"{'─'*55}\n")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    # Target the last residual block of ResNet-18 (layer4[-1])
    # This layer has the most abstract, semantically meaningful feature maps
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model=model, target_layer=target_layer)

    # Re-run with gradients enabled for Grad-CAM (predict_image used no_grad)
    tensor_for_cam = pil_to_tensor(pil_img, img_size=img_size).to(device)
    tensor_for_cam.requires_grad_(True)
    cam = gradcam.generate(tensor_for_cam, class_idx=pred_idx)
    gradcam.remove_hooks()

    # ── Prepare display images ────────────────────────────────────────────────
    # Convert PIL to OpenCV BGR for overlay
    img_rgb  = np.array(pil_img.resize((img_size, img_size)))  # (H, W, 3) RGB
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay  = overlay_heatmap_on_image(img_bgr, cam, alpha=0.45)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # ── Create and save side-by-side figure ───────────────────────────────────
    colour = "red" if pred_label == "FAKE" else "green"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=13, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title("Grad-CAM Explanation\n(red = high attention)", fontsize=13)
    axes[1].axis("off")

    verdict = f"Prediction: {pred_label}  (confidence: {confidence*100:.1f}%)"
    fig.suptitle(verdict, fontsize=14, fontweight='bold', color=colour, y=1.01)

    # Add a small colorbar legend
    import matplotlib.cm as cm
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Attention", fontsize=10)

    plt.tight_layout()

    # Determine output path
    if args.out is None:
        os.makedirs("outputs", exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.image))[0]
        out_path = f"outputs/{basename}_gradcam_{pred_label.lower()}.png"
    else:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Result image saved → {out_path}")
    print(f"\n  Summary: This image is classified as {pred_label} "
          f"with {confidence*100:.1f}% confidence.")
    print(f"  The Grad-CAM overlay highlights the face regions that most")
    print(f"  influenced the model's decision.\n")


if __name__ == "__main__":
    main()
