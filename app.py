"""
app.py
======
Gradio web UI for AI-Based Deepfake Detection.

Features:
  - Drag-and-drop / click-to-upload image input
  - Step-by-step progress bar showing exactly what's happening
  - Grad-CAM heatmap overlay displayed in the UI
  - Prediction label (REAL / FAKE) with colour-coded badge
  - Confidence score, FAKE probability, REAL probability
  - Error messages shown inline (no crashes)

Run:
    python app.py

Then open  http://127.0.0.1:7860  in your browser.
"""

import os
import io
import traceback

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
from torchvision import models

from gradcam_utils import GradCAM, overlay_heatmap_on_image, pil_to_tensor


# ─── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT  = "checkpoints/model.pth"
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Load model once at startup ───────────────────────────────────────────────
def load_model():
    """Load the trained ResNet-18 checkpoint. Returns (model, class_to_idx) or None."""
    if not os.path.isfile(CHECKPOINT):
        return None, None

    ckpt         = torch.load(CHECKPOINT, map_location=DEVICE)
    class_to_idx = ckpt.get("class_to_idx", {"fake": 0, "real": 1})

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model, class_to_idx


MODEL, CLASS_TO_IDX = load_model()


# ─── Core inference function ──────────────────────────────────────────────────
def analyze_image(pil_image: Image.Image, progress=gr.Progress()):
    """
    Full pipeline: preprocess → predict → Grad-CAM → return results.
    progress() calls update the Gradio progress bar in real time.

    Returns:
        heatmap_img   : PIL Image — original + heatmap side by side
        verdict_html  : HTML string — coloured prediction badge
        metrics_html  : HTML string — confidence / fake prob / real prob table
        status_text   : str — final status message
    """

    # ── Step 0: Guard checks ──────────────────────────────────────────────────
    progress(0.0, desc="⏳ Starting analysis...")

    if pil_image is None:
        return None, "", "", "❌ No image uploaded. Please upload a face image."

    if MODEL is None:
        return (
            None, "", "",
            "❌ Model not found.\n"
            "Please run  python train.py  first to train the model,\n"
            f"then make sure '{CHECKPOINT}' exists."
        )

    try:
        # ── Step 1: Load & preprocess ─────────────────────────────────────────
        progress(0.15, desc="🔍 Step 1/4 — Preprocessing image...")

        pil_image = pil_image.convert("RGB")
        tensor    = pil_to_tensor(pil_image, img_size=IMG_SIZE).to(DEVICE)

        # ── Step 2: Forward pass (inference) ─────────────────────────────────
        progress(0.35, desc="🧠 Step 2/4 — Running model inference...")

        fake_idx = CLASS_TO_IDX.get("fake", 0)
        real_idx = CLASS_TO_IDX.get("real", 1)

        with torch.no_grad():
            logits = MODEL(tensor)
            probs  = F.softmax(logits, dim=1)   # (1, 2)

        pred_idx   = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
        fake_prob  = probs[0, fake_idx].item()
        real_prob  = probs[0, real_idx].item()
        pred_label = "FAKE" if pred_idx == fake_idx else "REAL"

        # ── Step 3: Grad-CAM ──────────────────────────────────────────────────
        progress(0.60, desc="🌡️  Step 3/4 — Generating Grad-CAM heatmap...")

        target_layer = MODEL.layer4[-1]
        gradcam      = GradCAM(model=MODEL, target_layer=target_layer)

        # Need a fresh tensor with grad tracking for the backward pass
        tensor_cam = pil_to_tensor(pil_image, img_size=IMG_SIZE).to(DEVICE)
        tensor_cam.requires_grad_(True)
        cam = gradcam.generate(tensor_cam, class_idx=pred_idx)
        gradcam.remove_hooks()

        # ── Step 4: Build output image ────────────────────────────────────────
        progress(0.80, desc="🎨 Step 4/4 — Rendering result image...")

        orig_np     = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))
        orig_bgr    = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        overlay_bgr = overlay_heatmap_on_image(orig_bgr, cam, alpha=0.45)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # Side-by-side figure
        colour      = "#e74c3c" if pred_label == "FAKE" else "#2ecc71"
        fig, axes   = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor("#1a1a2e")

        axes[0].imshow(orig_np)
        axes[0].set_title("Original Image", fontsize=13,
                           fontweight='bold', color='white', pad=8)
        axes[0].axis("off")
        for spine in axes[0].spines.values():
            spine.set_edgecolor(colour); spine.set_linewidth(2)

        axes[1].imshow(overlay_rgb)
        axes[1].set_title("Grad-CAM Heatmap\n(red = high attention)",
                           fontsize=13, fontweight='bold', color='white', pad=8)
        axes[1].axis("off")
        for spine in axes[1].spines.values():
            spine.set_edgecolor(colour); spine.set_linewidth(2)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='jet',
                                    norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Attention Level", fontsize=10, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        plt.tight_layout(pad=2)

        # Convert figure to PIL image (so Gradio can display it)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130,
                    bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        result_img = Image.open(buf).copy()
        buf.close()

        # ── Build HTML outputs ────────────────────────────────────────────────
        progress(0.95, desc="✅ Finalising results...")

        # Verdict badge
        icon         = "🔴" if pred_label == "FAKE" else "🟢"
        verdict_html = f"""
        <div style="
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            background: {'#3d0000' if pred_label == 'FAKE' else '#003d00'};
            border: 3px solid {colour};
            margin: 8px 0;
        ">
            <div style="font-size: 48px; margin-bottom: 8px;">{icon}</div>
            <div style="
                font-size: 36px;
                font-weight: 900;
                color: {colour};
                letter-spacing: 4px;
            ">{pred_label}</div>
            <div style="
                font-size: 18px;
                color: #cccccc;
                margin-top: 6px;
            ">Prediction Result</div>
        </div>
        """

        # Metrics table
        conf_bar   = int(confidence * 100)
        fake_bar   = int(fake_prob * 100)
        real_bar   = int(real_prob * 100)

        def bar_html(pct, bar_colour):
            return (
                f'<div style="background:#333;border-radius:6px;height:14px;margin-top:4px;">'
                f'<div style="width:{pct}%;background:{bar_colour};height:100%;'
                f'border-radius:6px;transition:width 0.5s;"></div></div>'
            )

        metrics_html = f"""
        <div style="
            padding: 16px;
            border-radius: 10px;
            background: #1e1e2e;
            color: #f0f0f0;
            font-family: monospace;
            font-size: 15px;
        ">
            <div style="margin-bottom:14px;">
                <span style="color:#aaa;">Confidence</span>
                <span style="float:right;font-weight:bold;color:{colour};">
                    {confidence*100:.2f}%
                </span>
                {bar_html(conf_bar, colour)}
            </div>
            <div style="margin-bottom:14px;">
                <span style="color:#aaa;">FAKE Probability</span>
                <span style="float:right;font-weight:bold;color:#e74c3c;">
                    {fake_prob*100:.2f}%
                </span>
                {bar_html(fake_bar, '#e74c3c')}
            </div>
            <div>
                <span style="color:#aaa;">REAL Probability</span>
                <span style="float:right;font-weight:bold;color:#2ecc71;">
                    {real_prob*100:.2f}%
                </span>
                {bar_html(real_bar, '#2ecc71')}
            </div>
        </div>
        """

        progress(1.0, desc="✅ Done!")
        status = (
            f"✅ Analysis complete — "
            f"Predicted: {pred_label} | "
            f"Confidence: {confidence*100:.2f}% | "
            f"Device: {str(DEVICE).upper()}"
        )
        return result_img, verdict_html, metrics_html, status

    except Exception as e:
        err_detail = traceback.format_exc()
        error_msg  = (
            f"❌ Error during analysis:\n\n"
            f"{type(e).__name__}: {e}\n\n"
            f"Details:\n{err_detail}"
        )
        return None, "", "", error_msg


# ─── Build Gradio UI ──────────────────────────────────────────────────────────
def build_ui():
    model_status = (
        f"✅ Model loaded — Device: **{str(DEVICE).upper()}**"
        if MODEL is not None
        else "⚠️ Model not loaded — run `python train.py` first, then restart this app."
    )

    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="red",
            neutral_hue="slate",
        ),
        title="Deepfake Detection",
        css="""
        .gradio-container { max-width: 1100px !important; margin: auto; }
        #title { text-align: center; margin-bottom: 4px; }
        #subtitle { text-align: center; color: #888; margin-bottom: 20px; }
        #status_box textarea {
            font-family: monospace !important;
            font-size: 13px !important;
        }
        """
    ) as demo:

        # Header
        gr.HTML("""
        <div id="title">
            <h1 style="font-size:2.2em; font-weight:900; margin:0;">
                🕵️ AI Deepfake Detection
            </h1>
        </div>
        <div id="subtitle">
            <p style="font-size:1.05em;">
                Upload a face image → get a <b>REAL / FAKE</b> prediction
                with a <b>Grad-CAM</b> explanation heatmap
            </p>
        </div>
        """)

        gr.Markdown(f"> {model_status}")

        with gr.Row():
            # ── Left column: input ────────────────────────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="📁 Upload Face Image",
                    image_mode="RGB",
                    height=300,
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg",
                )
                gr.Examples(
                    examples=[
                        [f"data/real/{f}"]
                        for f in os.listdir("data/real")
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ][:3] +
                    [
                        [f"data/fake/{f}"]
                        for f in os.listdir("data/fake")
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ][:3]
                    if (os.path.isdir("data/real") and os.path.isdir("data/fake"))
                    else [],
                    inputs=image_input,
                    label="Quick Examples (from your dataset)",
                )

            # ── Right column: outputs ─────────────────────────────────────────
            with gr.Column(scale=2):
                verdict_out = gr.HTML(label="Prediction")
                metrics_out = gr.HTML(label="Probabilities")

        # Heatmap output (full width)
        heatmap_out = gr.Image(
            type="pil",
            label="🌡️ Original Image  |  Grad-CAM Heatmap",
            height=380,
            interactive=False,
        )

        # Status / progress log
        status_out = gr.Textbox(
            label="📋 Status Log",
            lines=3,
            interactive=False,
            elem_id="status_box",
            placeholder="Status messages will appear here during analysis...",
        )

        # ── How it works accordion ────────────────────────────────────────────
        with gr.Accordion("ℹ️ How it works", open=False):
            gr.Markdown("""
            **Model:** ResNet-18 pretrained on ImageNet, fine-tuned with a binary
            classification head (REAL / FAKE).

            **Grad-CAM Explanation:**
            1. The image is passed through the CNN — the last convolutional layer's
               feature maps are captured.
            2. Gradients of the predicted class score are backpropagated to that layer.
            3. Each feature map is weighted by its average gradient (importance).
            4. The weighted sum is upsampled and overlaid as a heatmap.
            5. **Red/yellow regions** = where the model focused most to make its decision.

            **Confidence** = probability of the predicted class.
            **FAKE Probability** = raw softmax score for the FAKE class.
            **REAL Probability** = raw softmax score for the REAL class.
            """)

        # ── Wire up the button ────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input],
            outputs=[heatmap_out, verdict_out, metrics_out, status_out],
        )

        # Also trigger on image upload (optional — remove if you prefer manual trigger)
        image_input.upload(
            fn=analyze_image,
            inputs=[image_input],
            outputs=[heatmap_out, verdict_out, metrics_out, status_out],
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Deepfake Detection — Web UI")
    print("=" * 55)
    print(f"  Device : {DEVICE}")
    print(f"  Model  : {'LOADED ✓' if MODEL is not None else 'NOT FOUND — run train.py first'}")
    print(f"  URL    : http://127.0.0.1:7860")
    print("=" * 55 + "\n")

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,          # set True to get a public Gradio link
        show_error=True,
    )
