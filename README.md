<div align="center">

<h1>🤟 Real-Time ASL Sign Language to Text Conversion</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Real--Time-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p>
  <strong>A deep learning–powered web application that translates American Sign Language (ASL) hand gestures into text in real time using your webcam.</strong><br/>
  Built with EfficientNet-B0, Flask, and OpenCV — bridging the gap between the Deaf community and the hearing world.
</p>

<br/>

> 🎯 **29 ASL classes** · ⚡ **Real-time inference** · 🌐 **Browser-based UI** · 🧠 **EfficientNet-B0 backbone**

</div>

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Demo & Features](#-demo--features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [How It Works](#-how-it-works)
- [Model Details](#-model-details)
- [Dataset](#-dataset)
- [Results](#-results)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🌟 About the Project

American Sign Language (ASL) is the primary language of millions of Deaf and hard-of-hearing individuals in the United States. Despite its prevalence, real-time translation tools remain inaccessible to most people.

This project addresses that gap by building a **live, webcam-based ASL recognition system** that:
- Detects hand signs in real time through a browser interface
- Converts recognized gestures into on-screen text, letter by letter
- Supports full sentence construction with **space** and **delete** gesture commands
- Runs on both CPU and GPU (CUDA-accelerated when available)

The system is designed to be **lightweight, fast, and easy to deploy** — no external APIs or cloud services required.

---

## 🎬 Demo & Features

| Feature | Description |
|---|---|
| 🎥 **Live Webcam Feed** | Real-time video stream rendered directly in the browser |
| 🤖 **EfficientNet-B0 Classifier** | High-accuracy, low-latency sign recognition |
| 🔤 **Sentence Builder** | Accumulates letters into words and sentences over time |
| 🗑️ **Delete Gesture** | Sign `del` to remove the last character |
| 🔲 **Space Gesture** | Sign `space` to insert a word gap |
| 📊 **Top-5 Confidence Display** | Shows top-5 class probabilities live |
| 🔄 **Temporal Smoothing** | 7-frame sliding window reduces flickering predictions |
| ⏯️ **Toggle Detection** | Pause/resume inference without closing the app |
| 📡 **REST API** | Clean JSON endpoints for state, clear, and toggle |
| ⚡ **CUDA / CPU Auto-select** | Automatically uses GPU if available |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (UI)                         │
│   Live Video Stream  │  Prediction Panel  │  Sentence Box   │
└──────────────┬─────────────────┬──────────────────┬─────────┘
               │                 │                  │
               ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Backend (app.py)                    │
│                                                             │
│  /video_feed  ──►  OpenCV Webcam  ──►  ROI Extraction       │
│                                           │                 │
│                                           ▼                 │
│                                   EfficientNet-B0           │
│                                   (PyTorch Inference)       │
│                                           │                 │
│                                           ▼                 │
│  /state  ◄──────  Temporal Smoothing  ◄──  Softmax Probs    │
│  /toggle_detection                        │                 │
│  /clear_sentence                          ▼                 │
│                                   Sentence Builder          │
└─────────────────────────────────────────────────────────────┘
```

**Inference Pipeline:**
1. Webcam frame is captured and horizontally flipped (mirror mode)
2. A fixed ROI (Region of Interest) box is cropped from the frame
3. ROI is converted to grayscale → resized to 224×224 → normalized
4. EfficientNet-B0 runs forward pass → softmax probabilities
5. A 7-frame sliding window averages probabilities (temporal smoothing)
6. Prediction with confidence ≥ 60% is accepted; held for 7 frames before appending to sentence

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Deep Learning** | PyTorch, TorchVision, EfficientNet-B0 |
| **Computer Vision** | OpenCV, PIL/Pillow |
| **Web Framework** | Flask, Flask-CORS |
| **Data Science** | TensorFlow, scikit-learn, pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter Notebook |
| **Language** | Python 3.8+ |

---

## 📁 Project Structure

```
Real-time-Sign-Language-to-Text-Conversion/
│
├── app.py                    # 🚀 Main Flask application & inference engine
├── requirements.txt          # 📦 Python dependencies
│
├── notebooks_2/              # 📓 Training & experimentation notebooks
│   └── outputs/
│       ├── best_model.pth    # 🧠 Trained EfficientNet-B0 weights
│       └── class_names.json  # 🗂️ Index-to-class label mapping
│
├── data/                     # 📂 Dataset directory
│
├── models/                   # 🗃️ Saved model checkpoints
│
├── src/                      # 🔧 Source modules / helper scripts
│
├── templates/                # 🌐 HTML templates for Flask
│   └── index.html            # Main web UI
│
├── reports/                  # 📊 Project reports & analysis
│
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8 or higher
- A working webcam
- `pip` package manager
- *(Optional)* NVIDIA GPU with CUDA for faster inference

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/HeshMaddage/Real-time-Sign-Language-to-Text-Conversion.git
cd Real-time-Sign-Language-to-Text-Conversion
```

**2. Create a virtual environment** *(recommended)*

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> 💡 **GPU Users:** If you want CUDA-accelerated inference, install the CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above command.

**4. Verify the model file exists**

Ensure the trained model checkpoint is at:
```
notebooks_2/outputs/best_model.pth
notebooks_2/outputs/class_names.json
```

### Running the App

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

The terminal will show:
```
✓ Class map loaded — 29 classes
Loading model on CUDA...   (or CPU)
✓ Model loaded (epoch=XX, val_acc=XX.XX%)

══════════════════════════════════════════════════
 ASL Detection Web App
 Open → http://localhost:5000
══════════════════════════════════════════════════
```

---

## 🔍 How It Works

### 1. Hand Gesture Capture
OpenCV captures frames from your webcam in real time. The frame is mirrored for a natural feel. A rectangular ROI box is displayed on screen — position your hand inside this box.

### 2. Preprocessing
Each ROI frame goes through:
- BGR → RGB → Grayscale (replicated to 3 channels)
- Resize to 224 × 224
- ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)

### 3. Deep Learning Inference
The preprocessed tensor is passed through **EfficientNet-B0** with a custom classifier head:
```
Dropout(0.3) → Linear(1280→512) → ReLU → Dropout(0.2) → Linear(512→29)
```
Softmax is applied to produce per-class probabilities across 29 ASL classes.

### 4. Temporal Smoothing
A **sliding window of 7 frames** averages the probability vectors to reduce flickering and improve prediction stability. The class with the highest averaged probability is selected.

### 5. Confidence Thresholding
Only predictions with confidence ≥ **60%** are acted upon. This prevents noisy or ambiguous frames from corrupting the output.

### 6. Sentence Building
A **hold mechanism** requires the same letter to be predicted for **7 consecutive frames** before it is appended to the sentence — preventing accidental duplicate characters.

Special gestures:
- **`space`** → inserts a space character
- **`del`** → removes the last character
- **`nothing`** → ignored (used for neutral/background frames)

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Architecture** | EfficientNet-B0 |
| **Input Size** | 224 × 224 × 3 |
| **Output Classes** | 29 (A–Z, `space`, `del`, `nothing`) |
| **Classifier Head** | Dropout → FC(512) → ReLU → Dropout → FC(29) |
| **Inference Device** | CUDA (GPU) / CPU auto-select |
| **Smoothing Window** | 7 frames |
| **Confidence Threshold** | 60% |
| **Hold Frames** | 7 frames |
| **Framework** | PyTorch |

The model is loaded from a checkpoint saved with:
```python
{
  "model_state": state_dict,
  "epoch": int,
  "val_acc": float
}
```

---

## 📊 Dataset

The project uses the **ASL Alphabet Dataset**, which contains:
- **29 classes**: Letters A–Z, plus `space`, `del`, and `nothing`
- Thousands of labeled hand gesture images per class
- Images captured under varying lighting and backgrounds

> The dataset is stored in the `data/` directory. You can use the [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) or similar sources.

---

## 📈 Results

The trained EfficientNet-B0 model achieves high accuracy on the ASL alphabet classification task.

| Metric | Value |
|---|---|
| **Validation Accuracy** | Stored in checkpoint (`val_acc` field) |
| **Classes** | 29 |
| **Real-time FPS** | Displayed live in the web UI |
| **Inference Latency** | Low (GPU) / moderate (CPU) |

> Training curves, confusion matrices, and detailed performance reports can be found in the `reports/` directory and the `notebooks_2/` Jupyter notebooks.

---


## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) — Tan & Le, Google Brain
- [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [PyTorch](https://pytorch.org/) & [TorchVision](https://pytorch.org/vision/)
- [OpenCV](https://opencv.org/) for computer vision utilities
- [Flask](https://flask.palletsprojects.com/) for the web backend
- The Deaf and hard-of-hearing community, whose needs inspire this work 💙

---

<div align="center">

Made with ❤️ by [HeshMaddage](https://github.com/HeshMaddage) and contributors

⭐ **If you find this project useful, please give it a star!** ⭐

</div>