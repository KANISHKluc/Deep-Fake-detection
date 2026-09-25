"""
gradcam_utils.py
================
Manual Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

HOW GRAD-CAM WORKS (for your report):
---------------------------------------
1. We pick a target convolutional layer (usually the last conv layer) whose
   feature maps still contain spatial information about the image.

2. We do a forward pass to get the model's prediction.

3. We backpropagate the score of the predicted class all the way back to
   the chosen conv layer — NOT to the input image.

4. For each of the K feature maps (channels) in that layer, we compute the
   AVERAGE gradient over the spatial dimensions (H x W). This gives us K
   scalar "importance weights" — one per feature map channel.
   α_k = (1/Z) * Σ_ij (∂y^c / ∂A^k_ij)
   where y^c is the class score, A^k is the k-th feature map.

5. We take a weighted sum of the feature maps using those importance weights,
   then apply ReLU to keep only the positive contributions:
   L_Grad-CAM = ReLU( Σ_k α_k * A^k )

6. The resulting map is resized to the input image size and overlaid as a
   heatmap. Hot regions (red) are where the model "looked" to make its
   decision — the explanation.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation using PyTorch hooks.

    Hooks let us intercept and store intermediate tensors during the
    forward and backward passes without modifying the model architecture.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model:        The trained PyTorch model (in eval mode).
            target_layer: The specific conv layer to visualize.
                          For ResNet-18 this is model.layer4[-1].
        """
        self.model = model
        self.target_layer = target_layer

        # These will be populated by the hooks during forward/backward passes
        self._activations: torch.Tensor = None  # forward feature maps A^k
        self._gradients: torch.Tensor = None    # gradients ∂y^c/∂A^k

        # Register hooks on the target layer
        self._forward_hook = target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        """Forward hook: captures the feature maps A^k from the target layer."""
        # output shape: (batch, channels, H, W)
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook: captures ∂y^c/∂A^k flowing back through the target layer."""
        # grad_output[0] shape: (batch, channels, H, W)
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor, shape (1, C, H, W).
            class_idx:    Which class to explain. If None, uses the predicted class.

        Returns:
            cam: Normalized heatmap as a float32 numpy array, values in [0, 1],
                 shape (H_orig, W_orig) — same spatial size as the input.
        """
        self.model.eval()

        # ── Step 1: Forward pass ─────────────────────────────────────────────
        # The forward hook fires here, saving activations A^k
        output = self.model(input_tensor)  # shape: (1, num_classes)

        # ── Step 2: Choose target class ──────────────────────────────────────
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # ── Step 3: Backward pass for the target class score ─────────────────
        self.model.zero_grad()
        # Create a one-hot gradient signal: only the target class score flows back
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        # The backward hook fires here, saving gradients ∂y^c/∂A^k
        output.backward(gradient=one_hot)

        # ── Step 4: Compute per-channel importance weights α_k ───────────────
        # Global Average Pooling over spatial dims (H, W) → shape: (1, channels)
        # This averages the gradients over all spatial positions, giving a
        # single scalar importance weight per feature map channel.
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, K, 1, 1)

        # ── Step 5: Weighted combination of forward activations ───────────────
        # Multiply each feature map A^k by its weight α_k, then sum across channels
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')

        # ReLU: keep only features that positively influence the class prediction
        cam = F.relu(cam)

        # ── Step 6: Resize to input image dimensions ─────────────────────────
        _, _, H, W = input_tensor.shape
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)

        # Normalize to [0, 1] for visualization
        cam = cam.squeeze().cpu().numpy()  # (H, W)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)  # flat map → no meaningful explanation

        return cam.astype(np.float32)

    def remove_hooks(self):
        """Clean up hooks to free memory and avoid duplicate hook accumulation."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def overlay_heatmap_on_image(
    original_image: np.ndarray,
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend the Grad-CAM heatmap onto the original image.

    Args:
        original_image: BGR or RGB uint8 image, shape (H, W, 3).
        cam:            Float32 heatmap in [0, 1], shape (H, W).
        colormap:       OpenCV colormap constant (default: JET — blue→red).
        alpha:          Heatmap opacity (0 = invisible, 1 = fully opaque).

    Returns:
        blended: uint8 BGR image with the heatmap overlaid.
    """
    # Scale cam to 0–255 and apply a false-colour map (JET: blue=cold, red=hot)
    heatmap_uint8 = np.uint8(255 * cam)          # (H, W) in 0–255
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)  # (H, W, 3) BGR

    # Ensure the original image is in BGR for OpenCV blending
    if original_image.shape[2] == 3:
        img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR) \
                  if original_image.dtype == np.uint8 else original_image
    else:
        img_bgr = original_image

    # Resize heatmap to match original image size (should already match, but safety check)
    if heatmap_color.shape[:2] != img_bgr.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (img_bgr.shape[1], img_bgr.shape[0]))

    # Alpha-blend: blended = alpha * heatmap + (1 - alpha) * image
    blended = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)
    return blended


def pil_to_tensor(image: Image.Image, img_size: int = 224) -> torch.Tensor:
    """
    Convert a PIL image to a normalised model-ready tensor.

    Uses standard ImageNet mean/std normalisation, since the backbone
    was pretrained on ImageNet.

    Args:
        image:    PIL Image (RGB).
        img_size: Resize target (224 for ResNet/EfficientNet).

    Returns:
        Tensor of shape (1, 3, img_size, img_size).
    """
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean (R, G, B)
            std=[0.229, 0.224, 0.225],   # ImageNet std  (R, G, B)
        ),
    ])
    return preprocess(image).unsqueeze(0)  # add batch dimension → (1, 3, H, W)
