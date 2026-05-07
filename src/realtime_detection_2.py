import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import deque
import numpy as np
import json
import time

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION — edit these paths before running
# ═══════════════════════════════════════════════════════════
MODEL_PATH      = "notebooks_2/outputs/best_model.pth"
CLASS_MAP_PATH  = "notebooks_2/outputs/class_names.json"
NUM_CLASSES     = 29
IMG_SIZE        = 224
CONFIDENCE_THRESHOLD = 0.60   # only show prediction above this confidence
SMOOTHING_WINDOW     = 7      # number of frames to average predictions over
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════
#  LOAD CLASS NAMES
# ═══════════════════════════════════════════════════════════
try:
    with open(CLASS_MAP_PATH, "r") as f:
        raw = json.load(f)
    # Keys are strings ("0", "1", ...) — convert to int
    idx_to_class = {int(k): v for k, v in raw.items()}
    print(f"✓ Class map loaded — {len(idx_to_class)} classes")
except Exception as e:
    print(f"✗ Could not load class map: {e}")
    idx_to_class = {i: str(i) for i in range(NUM_CLASSES)}

# ═══════════════════════════════════════════════════════════
#  BUILD MODEL — must match training architecture exactly
# ═══════════════════════════════════════════════════════════
print("Loading model...")

model = models.efficientnet_b0(weights=None)

# Rebuild the exact same classifier head used during training:
#   Dropout(0.3) → Linear(1280, 512) → ReLU → Dropout(0.2) → Linear(512, 29)
in_features = model.classifier[1].in_features   # 1280 for EfficientNet-B0
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(512, NUM_CLASSES),
)

# Load the saved checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()

saved_epoch   = checkpoint.get("epoch", "?")
saved_val_acc = checkpoint.get("val_acc", 0) * 100
print(f"✓ Model loaded (epoch {saved_epoch}, val_acc={saved_val_acc:.2f}%)")

# ═══════════════════════════════════════════════════════════
#  INFERENCE TRANSFORM — must match val_transform from training
#  (grayscale → 3-channel, resize, normalise with ImageNet stats)
# ═══════════════════════════════════════════════════════════
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),  # colour-invariant, matches training
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                         # [0,255] → [0.0,1.0]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ═══════════════════════════════════════════════════════════
#  TEMPORAL SMOOTHER
#  Averages softmax probabilities over the last N frames so the
#  prediction doesn't flicker when the hand is between gestures.
# ═══════════════════════════════════════════════════════════
prob_buffer = deque(maxlen=SMOOTHING_WINDOW)

def smooth_predict(logits: torch.Tensor):
    """
    Given raw logits (shape: [1, num_classes]), return:
      - smoothed predicted class index (int)
      - smoothed confidence for that class (float 0-1)
    """
    probs = torch.softmax(logits.squeeze(0).cpu(), dim=-1).numpy()
    prob_buffer.append(probs)
    avg_probs   = np.mean(list(prob_buffer), axis=0)
    pred_idx    = int(np.argmax(avg_probs))
    confidence  = float(avg_probs[pred_idx])
    return pred_idx, confidence

# ═══════════════════════════════════════════════════════════
#  OVERLAY HELPERS
# ═══════════════════════════════════════════════════════════
def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, radius=12):
    """Draw a rectangle with rounded corners."""
    cv2.line(img,  (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(img,  (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(img,  (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(img,  (x2, y1+radius), (x2, y2-radius), color, thickness)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius,radius), 180, 0,  90,  color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius,radius), 270, 0,  90,  color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius,radius),  90, 0,  90,  color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius,radius),   0, 0,  90,  color, thickness)

def draw_confidence_bar(img, x, y, width, confidence, color):
    """Draw a filled confidence bar."""
    bar_h   = 10
    filled  = int(width * confidence)
    cv2.rectangle(img, (x, y), (x+width, y+bar_h), (60, 60, 60), -1)        # background
    cv2.rectangle(img, (x, y), (x+filled, y+bar_h), color, -1)              # filled portion
    cv2.rectangle(img, (x, y), (x+width, y+bar_h), (120, 120, 120), 1)      # border

def put_text_with_shadow(img, text, pos, font_scale, color, thickness=2):
    """Draw text with a dark shadow for readability on any background."""
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════
#  MAIN WEBCAM LOOP
# ═══════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check your camera index.")

# FPS tracking
fps_counter = 0
fps_start   = time.time()
fps_display = 0.0

print("\n✓ Webcam started")
print("  Place your hand in the GREEN box")
print("  Press  Q  to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame — exiting.")
        break

    # Mirror the frame (natural webcam feel)
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # ── ROI: square box on the right side of the frame ────────────────────────
    box_size   = min(300, h - 80, w - 80)
    roi_left   = w - box_size - 40
    roi_right  = roi_left + box_size
    roi_top    = (h - box_size) // 2
    roi_bottom = roi_top + box_size

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    # ── Preprocess & infer ────────────────────────────────────────────────────
    rgb_roi   = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor    = infer_transform(rgb_roi).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

    pred_idx, confidence = smooth_predict(logits)
    pred_label = idx_to_class.get(pred_idx, "?")

    # ── FPS calculation ───────────────────────────────────────────────────────
    fps_counter += 1
    elapsed = time.time() - fps_start
    if elapsed >= 1.0:
        fps_display  = fps_counter / elapsed
        fps_counter  = 0
        fps_start    = time.time()

    # ── Draw ROI box ──────────────────────────────────────────────────────────
    box_color = (0, 220, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 140, 255)
    draw_rounded_rect(frame, roi_left, roi_top, roi_right, roi_bottom, box_color, 2)
    put_text_with_shadow(frame, "Place hand here",
                         (roi_left + 8, roi_top - 10), 0.5, box_color, 1)

    # ── Prediction panel (top-left) ───────────────────────────────────────────
    panel_x, panel_y = 15, 15
    panel_w, panel_h = 280, 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x+panel_w, panel_y+panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)   # semi-transparent bg

    if confidence >= CONFIDENCE_THRESHOLD:
        sign_color = (0, 220, 0)
        sign_text  = pred_label
        conf_text  = f"{confidence*100:.1f}%"
    else:
        sign_color = (0, 140, 255)
        sign_text  = "..."
        conf_text  = f"{confidence*100:.1f}%  (low)"

    # Large predicted letter
    put_text_with_shadow(frame, sign_text,
                         (panel_x+14, panel_y+72), 2.4, sign_color, 3)

    # Confidence label + bar
    put_text_with_shadow(frame, f"Conf: {conf_text}",
                         (panel_x+14, panel_y+96), 0.48, (200, 200, 200), 1)
    draw_confidence_bar(frame,
                        panel_x+14, panel_y+103,
                        panel_w-28, confidence, sign_color)

    # ── Stats bar (bottom-left) ───────────────────────────────────────────────
    put_text_with_shadow(frame, f"FPS: {fps_display:.1f}",
                         (15, h-40), 0.5, (180, 180, 180), 1)
    put_text_with_shadow(frame, f"Device: {str(DEVICE).upper()}",
                         (15, h-18), 0.5, (180, 180, 180), 1)
    put_text_with_shadow(frame, "Q = quit",
                         (w-100, h-18), 0.5, (180, 180, 180), 1)

    cv2.imshow("ASL Real-Time Detection — EfficientNet-B0", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ═══════════════════════════════════════════════════════════
#  CLEANUP
# ═══════════════════════════════════════════════════════════
cap.release()
cv2.destroyAllWindows()
print("Done.")