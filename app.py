"""
ASL Real-Time Detection — Flask Backend
Run: python app.py
Then open: http://localhost:5000
"""

import cv2
import json
import time
import base64
import threading
import numpy as np
from collections import deque
from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════
MODEL_PATH     = "notebooks_2/outputs/best_model.pth"
CLASS_MAP_PATH = "notebooks_2/outputs/class_names.json"
NUM_CLASSES    = 29
IMG_SIZE       = 224
SMOOTHING_WINDOW      = 7
CONFIDENCE_THRESHOLD  = 0.60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════
#  LOAD CLASS NAMES
# ═══════════════════════════════════════════════════════════
with open(CLASS_MAP_PATH, "r") as f:
    raw = json.load(f)
idx_to_class = {int(k): v for k, v in raw.items()}
print(f"✓ Class map loaded — {len(idx_to_class)} classes")

# ═══════════════════════════════════════════════════════════
#  LOAD MODEL
# ═══════════════════════════════════════════════════════════
print(f"Loading model on {DEVICE}...")
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(512, NUM_CLASSES),
)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()
MODEL_EPOCH   = checkpoint.get("epoch", "?")
MODEL_VAL_ACC = checkpoint.get("val_acc", 0) * 100
print(f"✓ Model loaded (epoch={MODEL_EPOCH}, val_acc={MODEL_VAL_ACC:.2f}%)")

# ═══════════════════════════════════════════════════════════
#  INFERENCE TRANSFORM
# ═══════════════════════════════════════════════════════════
infer_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ═══════════════════════════════════════════════════════════
#  SHARED STATE (thread-safe via lock)
# ═══════════════════════════════════════════════════════════
state_lock = threading.Lock()
shared_state = {
    "label":       "—",
    "confidence":  0.0,
    "top5":        [],
    "fps":         0.0,
    "device":      str(DEVICE).upper(),
    "model_epoch": MODEL_EPOCH,
    "val_acc":     f"{MODEL_VAL_ACC:.1f}",
    "sentence":    [],
    "detection_on": True,
}

prob_buffer  = deque(maxlen=SMOOTHING_WINDOW)
last_letter  = None
letter_hold  = 0
HOLD_FRAMES  = 5   # frames to hold before appending to sentence

# ═══════════════════════════════════════════════════════════
#  INFERENCE FUNCTION
# ═══════════════════════════════════════════════════════════
def run_inference(roi_bgr):
    """Run model on an ROI (BGR numpy array). Returns (label, confidence, top5)."""
    rgb  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    tensor = infer_transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

    probs = torch.softmax(logits.squeeze(0).cpu(), dim=-1).numpy()
    prob_buffer.append(probs)
    avg = np.mean(list(prob_buffer), axis=0)

    pred_idx    = int(np.argmax(avg))
    confidence  = float(avg[pred_idx])
    label       = idx_to_class.get(pred_idx, "?")

    top5_indices = np.argsort(avg)[::-1][:5]
    top5 = [
        {"label": idx_to_class.get(int(i), "?"), "prob": float(avg[i])}
        for i in top5_indices
    ]
    return label, confidence, top5


# ═══════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

cap         = None
cam_lock    = threading.Lock()
fps_counter = 0
fps_start   = time.time()


def get_camera():
    global cap
    with cam_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
    return cap


def generate_frames():
    global fps_counter, fps_start, last_letter, letter_hold
    camera = get_camera()

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── ROI ──────────────────────────────────────────────────────────────
        box_size   = min(280, h - 60, w - 60)
        roi_left   = w - box_size - 30
        roi_right  = roi_left + box_size
        roi_top    = (h - box_size) // 2
        roi_bottom = roi_top + box_size

        roi = frame[roi_top:roi_bottom, roi_left:roi_right]

        # ── Inference ────────────────────────────────────────────────────────
        with state_lock:
            detection_on = shared_state["detection_on"]

        label, confidence, top5 = ("—", 0.0, [])
        if detection_on and roi.size > 0:
            try:
                label, confidence, top5 = run_inference(roi)
            except Exception:
                pass

        # ── Sentence builder ─────────────────────────────────────────────────
        with state_lock:
            if detection_on and confidence >= CONFIDENCE_THRESHOLD:
                if label == last_letter:
                    letter_hold += 1
                    if letter_hold == HOLD_FRAMES:
                        if label == "space":
                            shared_state["sentence"].append(" ")
                        elif label == "del":
                            if shared_state["sentence"]:
                                shared_state["sentence"].pop()
                        elif label != "nothing":
                            shared_state["sentence"].append(label)
                else:
                    last_letter  = label
                    letter_hold  = 0

            shared_state["label"]      = label
            shared_state["confidence"] = confidence
            shared_state["top5"]       = top5

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            with state_lock:
                shared_state["fps"] = round(fps_counter / elapsed, 1)
            fps_counter = 0
            fps_start   = time.time()

        # ── Draw minimal overlay on frame ─────────────────────────────────────
        # ROI box
        box_color = (0, 220, 80) if confidence >= CONFIDENCE_THRESHOLD else (0, 160, 255)
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), box_color, 2)

        # Encode and yield
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html",
                           device=str(DEVICE).upper(),
                           model_epoch=MODEL_EPOCH,
                           val_acc=f"{MODEL_VAL_ACC:.1f}")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def get_state():
    with state_lock:
        data = dict(shared_state)
        data["sentence_text"] = "".join(data["sentence"])
    return jsonify(data)


@app.route("/clear_sentence", methods=["POST"])
def clear_sentence():
    with state_lock:
        shared_state["sentence"] = []
    return jsonify({"ok": True})


@app.route("/toggle_detection", methods=["POST"])
def toggle_detection():
    with state_lock:
        shared_state["detection_on"] = not shared_state["detection_on"]
        status = shared_state["detection_on"]
    prob_buffer.clear()
    return jsonify({"detection_on": status})


if __name__ == "__main__":
    print("\n" + "═"*50)
    print("  ASL Detection Web App")
    print(f"  Open → http://localhost:5000")
    print("═"*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
