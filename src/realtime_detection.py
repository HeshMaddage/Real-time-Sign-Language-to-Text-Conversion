# import cv2
# import torch
# import numpy as np
# from torchvision import models, transforms
# import torch.nn as nn
# from PIL import Image
# import json
# import os

# # ================= LOAD CLASS MAPPING =================
# with open("data\processed\class_mapping.json", "r") as f:
#     class_mapping = json.load(f)["class_to_index"]

# idx_to_class = {v: k for k, v in class_mapping.items()}

# # ================= LOAD MODEL =================
# num_classes = len(idx_to_class)

# model = models.mobilenet_v2(weights=None)
# in_features = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(in_features, num_classes)

# model.load_state_dict(torch.load("best_model_finetuned.pth", map_location="cpu"))
# model.eval()

# print("Model loaded")

# # ================= TRANSFORM =================
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5],
#                          [0.5, 0.5, 0.5])
# ])

# # ================= WEBCAM =================
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Cannot open webcam")
#     exit()

# print("Press 'q' to quit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip for mirror effect
#     frame = cv2.flip(frame, 1)

#     # Convert BGR → RGB
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Convert to PIL Image
#     pil_img = Image.fromarray(rgb)

#     # Preprocess
#     img = transform(pil_img).unsqueeze(0)  # (1, 3, 128, 128)

#     # Prediction
#     with torch.no_grad():
#         outputs = model(img)
#         probs = torch.softmax(outputs, dim=1)
#         conf, pred = torch.max(probs, 1)

#     label = idx_to_class[pred.item()]
#     confidence = conf.item()

#     # Display prediction
#     text = f"{label} ({confidence:.2f})"

#     cv2.putText(frame, text, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1, (0, 255, 0), 2)

#     cv2.imshow("ASL Detection", frame)

#     # Quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import os

# ================= CONFIG =================
MODEL_PATH = "best_model_finetuned.pth"   # update if needed
CLASSES = [chr(i) for i in range(65, 91)]  # A-Z (dummy classes for testing)

# ================= LOAD MODEL =================
num_classes = len(CLASSES)

model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("Model loaded")
else:
    print("Model NOT found — using random weights (just for testing)")

model.eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ================= WEBCAM =================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip (mirror view)
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    pil_img = Image.fromarray(rgb)

    # Preprocess
    img = transform(pil_img).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    label = CLASSES[pred.item()]

    # Display
    cv2.putText(frame, f"Prediction: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Sign Detection (Test)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()