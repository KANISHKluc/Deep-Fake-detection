# 🕵️ AI-Based Deepfake Detection Using Explainable Deep Learning

A self-contained Python project that classifies face images as **REAL** or **FAKE**
using a fine-tuned ResNet-18 CNN, with **Grad-CAM** visual explanations and a
clean **Gradio web UI** — no command line needed for inference.

Built as a college mini-project demo. Runs fully on CPU (GPU auto-detected if available).

---

## 📸 Web UI Preview

> Upload a face image and get an instant prediction with a heatmap explanation.

[![Web UI Screenshot](docs/Screenshot 2026-09-25 181053.png")](https://github.com/KANISHKluc/Deep-Fake-detection/blob/84144f6e8a3b92e523d1420e601591f1a444e845/docs/Screenshot%202026-09-25%20181053.png)

> ⚠️ To add your own screenshot: run `python app.py`, take a screenshot of the browser,
> save it as `docs/ui_screenshot.png` in the project folder.

---

## ✨ Features

- 🔍 **Deepfake detection** — binary classifier (REAL / FAKE) on any face image
- 🌡️ **Grad-CAM heatmap** — visual explanation showing *where* the model looked
- 🖥️ **Gradio web UI** — drag-and-drop image upload, no command line needed
- 📊 **Step-by-step progress bar** in the UI during analysis
- 📈 **Evaluation reports** — confusion matrix, ROC curve, metrics table, sample grid
- ⚡ **CPU-friendly** — trains in ~2–5 min on a tiny dataset; GPU used automatically if available

---

## 🗂️ Project Structure

```
deepfake-detection/
│
├── data/
│   ├── real/              ← Place real face images here (.jpg / .png)
│   └── fake/              ← Place AI-generated / deepfake images here
│
├── checkpoints/           ← Saved model weights (auto-created by train.py)
├── outputs/               ← All generated plots and metrics (auto-created)
├── docs/                  ← Screenshots for README
│
├── app.py                 ← 🌐 Gradio web UI (main entry point)
├── train.py               ← 🏋️ Fine-tune ResNet-18 on your dataset
├── predict.py             ← 🔮 Single-image CLI inference + Grad-CAM
├── evaluate.py            ← 📊 Full evaluation: metrics + plots
├── gradcam_utils.py       ← 🧠 Grad-CAM implementation (with comments)
├── requirements.txt       ← Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10, 3.11, or 3.12** (recommended — PyTorch does not yet support 3.14)
- pip
- Git

> **Python 3.14 users:** PyTorch wheels don't exist for 3.14 yet.
> Create a virtual environment with Python 3.11:
> ```bash
> py -3.11 -m venv venv
> venv\Scripts\activate   # Windows
> source venv/bin/activate  # macOS/Linux
> ```

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detection.git
cd deepfake-detection
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

**For GPU support (NVIDIA):** Install the CUDA-enabled PyTorch first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
Check your CUDA version with `nvidia-smi` (top-right corner of the output).

---

### Step 3 — Add your images

Place face images in the two data folders:

```
data/
  real/    →  real face photos        (10–20 images minimum)
  fake/    →  AI-generated faces      (10–20 images minimum)
```

Supported formats: `.jpg`, `.jpeg`, `.png`

**Where to get images:**

| Source | Type | Notes |
|--------|------|-------|
| [This Person Does Not Exist](https://thispersondoesnotexist.com) | Fake | Refresh page = new face. Right-click → Save image |
| [Pexels Portraits](https://www.pexels.com/search/portrait/) | Real | Free stock photos |
| [140k Real and Fake Faces – Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | Both | Large dataset, pick a small subset |
| [Real and Fake Face Detection – Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) | Both | Good for demos |

---

### Step 4 — Train the model

```bash
python train.py
```

**With custom options:**
```bash
python train.py --epochs 20 --batch_size 8 --lr 0.0005
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data` | Root folder with `real/` and `fake/` subfolders |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `8` | Reduce to `4` if you have fewer than 10 images |
| `--lr` | `0.0005` | Learning rate |
| `--val_split` | `0.2` | Fraction held out for validation |
| `--save_path` | `checkpoints/model.pth` | Where to save the best model |

**Expected output:**
```
=======================================================
  Deepfake Detection — Training
=======================================================
  Device    : cpu
  Total images : 40  (fake: 20, real: 20)
  Train samples: 32  |  Val samples: 8

  Epoch  Train Loss  Train Acc   Val Loss   Val Acc
  --------------------------------------------------
      1      0.7102     0.5263     0.6891    0.6000
      5      0.5431     0.7500     0.5102    0.7500
     20      0.2341     0.9375     0.3210    0.8750 ★

  Training complete in 142.3s  |  Best val acc: 0.8750
  Model saved → checkpoints/model.pth
```

---

### Step 5 — Launch the Web UI

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

**What you'll see:**
- 📁 Drag-and-drop (or click) to upload any face image
- 🔍 Click **Analyze Image** (or it runs automatically on upload)
- Progress bar steps through: Preprocessing → Inference → Grad-CAM → Rendering
- 🔴 **FAKE** / 🟢 **REAL** verdict badge
- Confidence %, FAKE probability, REAL probability — each with a visual bar
- Side-by-side: Original image | Grad-CAM heatmap
- Status log showing each step (errors appear here, no crashes)

---

### Step 6 — Evaluate the model (optional, for report figures)

```bash
python evaluate.py
```

**Generates in `outputs/`:**

| File | Description |
|------|-------------|
| `metrics.txt` | Accuracy, Precision, Recall, F1, AUC-ROC |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC and optimal threshold |
| `gradcam_grid.png` | ⭐ Grid of 8 test images with Grad-CAM overlays |
| `training_curves.png` | Loss and accuracy per epoch |

---

### Step 7 — Single image CLI prediction (optional)

```bash
python predict.py --image data/fake/some_face.jpg
```

Output saved to `outputs/` automatically.

---

## 🧠 How Grad-CAM Works

Grad-CAM answers: *"Which part of the image made the model say FAKE?"*

```
Input Image
     │
     ▼
ResNet-18 Backbone  (frozen — ImageNet weights)
     │
   layer1 → layer2 → layer3 → layer4  ← target layer
     │
     ▼
Global Average Pool → 512-dim vector
     │
     ▼
Classification Head  (trained)
  Dropout → Linear(512→128) → ReLU → Linear(128→2)
     │
     ▼
  Softmax → P(REAL),  P(FAKE)
```

**Grad-CAM steps:**
1. Forward pass — capture feature maps `A^k` at `layer4`
2. Backprop the predicted class score — capture gradients `∂y^c/∂A^k`
3. Average each gradient map spatially → importance weight `α_k`
4. Weighted sum + ReLU: `L = ReLU(Σ α_k · A^k)`
5. Upsample to input size → overlay as heatmap

**Red = high attention** (where the model focused).
**Blue = low attention** (irrelevant regions).

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
Pillow>=10.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
numpy>=1.24.0
grad-cam>=1.4.8
tqdm>=4.65.0
gradio>=4.0.0
```

---

## 🛠️ Troubleshooting

| Error | Fix |
|-------|-----|
| `Model not found` in UI | Run `python train.py` first |
| `Only 0 images found` | Check images are inside `data/real/` and `data/fake/` (not `data/` directly) |
| `TypeError: ReduceLROnPlateau... verbose` | Already fixed in current code |
| `No matching distribution for torch` | You're on Python 3.14 — use Python 3.11 (see Step 2) |
| `CUDA: False` with NVIDIA GPU | Install CUDA PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| Low accuracy | Normal for <20 images. Try `--epochs 30` or add more images |
| UI shows blank heatmap | Image may not contain a face, or model needs more training |

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📚 References

- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017 — [arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 — [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs)
- [Gradio Documentation](https://www.gradio.app/docs)

---

## 📄 License

This project is for educational purposes. See [LICENSE](LICENSE) for details.
