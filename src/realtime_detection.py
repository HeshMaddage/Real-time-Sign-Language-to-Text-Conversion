import cv2
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import json

# ================= Configuration =================
MODEL_PATH = 'notebooks_2/outputs/best_model.pth'
MAPPING_PATH = 'notebooks_2/outputs/class_names.json'
NUM_CLASSES = 29
INPUT_SIZE = (224, 224) # Adjust if your training images had a different resolution
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # ================= Load Mapping ==================
# try:
#     with open(MAPPING_PATH, 'r') as f:
#         # Assuming mapping is like {"A": 0, "B": 1...}, we need to reverse it to {0: "A", 1: "B"...}
#         class_mapping = json.load(f)["class_to_index"]
#         idx_to_class = {v: k for k, v in class_mapping.items()}
# except Exception as e:
#     print(f"Warning: Could not load class mapping. Error: {e}")
#     # Fallback to string indices if missing
#     idx_to_class = {i: str(i) for i in range(NUM_CLASSES)}

# ================= Load Mapping ==================
try:
    with open(MAPPING_PATH, 'r') as f:
        loaded_mapping = json.load(f)

        # Convert string keys to integer keys
        idx_to_class = {int(k): v for k, v in loaded_mapping.items()}

    print("Class mapping loaded successfully!")

except Exception as e:
    print(f"Warning: Could not load class mapping. Error: {e}")

    # Fallback labels
    idx_to_class = {i: str(i) for i in range(NUM_CLASSES)}

# # ================= Load Model ====================
# print("Loading model...")
# model = models.mobilenet_v2(pretrained=False) # No need to download pretrained weights again
# model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model = model.to(DEVICE)
# model.eval() # Set to evaluation mode
# print("Model loaded successfully!")

# ================= Load Model ====================
print("Loading model...")

# Create EfficientNet model
model = models.efficientnet_b0(weights=None)

# Replace final classification layer
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Load trained weights
model.load_state_dict(checkpoint["model_state"])

# Move to device
model = model.to(DEVICE)

# Evaluation mode
model.eval()

print("Model loaded successfully!")

# ================= Helper Function ===============
def preprocess_frame(frame):
    # 1. Resize the frame to match what the model was trained on
    img = cv2.resize(frame, INPUT_SIZE)
    
    # # 2. Convert BGR (OpenCV default) to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Convert BGR (OpenCV default) to Grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Duplicate the grayscale channel to 3 channels (MobileNetV2 expects 3 channels)
    img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    
    # 4. Scale pixel values. 
    # In your notebook you used: plt.imshow((X_train[0] + 1) / 2)
    # This implies your training data was scaled between [-1, 1].
    # So we map [0, 255] -> [0, 1] -> [-1, 1]
    img = img.astype(np.float32) / 255.0
    img = (img * 2.0) - 1.0 
    
    # 5. Format for PyTorch (C, H, W) and add batch dimension (1, C, H, W)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.tensor(img).unsqueeze(0).float()
    return tensor

# ================= Webcam Loop ===================
cap = cv2.VideoCapture(0) # 0 is usually the default webcam

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Optional: Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Define a Region of Interest (ROI) where the user should place their hand
    # Let's use a 300x300 box on the right side of the screen
    h, w, c = frame.shape
    roi_top, roi_bottom = 50, 350
    roi_left, roi_right = w - 350, w - 50
    
    # Draw the ROI box on the frame
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    
    # Extract the ROI and preprocess it
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    input_tensor = preprocess_frame(roi).to(DEVICE)
    
    # Run Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        # Apply softmax to get confidence scores
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        max_prob, predicted_idx = torch.max(probabilities, 0)
        
        confidence = max_prob.item() * 100
        predicted_label = idx_to_class.get(predicted_idx.item(), "Unknown")
    
    # Only display the prediction if confidence is high enough (e.g., > 60%)
    if confidence > 60:
        display_text = f"Sign: {predicted_label} ({confidence:.1f}%)"
    else:
        display_text = "Sign: Waiting..."
        
    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the final frame
    cv2.imshow('Real-time Sign Language Translator', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()