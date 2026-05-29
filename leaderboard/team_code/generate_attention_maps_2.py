#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp
from torchvision.models import ResNet50_Weights

# ------------------------------
# 1️⃣ Set Device to CPU
# ------------------------------
device = torch.device("cpu")
print(f"[INFO] Using device: {device}")

# ------------------------------
# 2️⃣ Load Pretrained ResNet50 Model
# ------------------------------
print("[INFO] Loading pretrained ResNet50 model...")
# Use the recommended weights from torchvision
weights = ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)
model.to(device)
model.eval()

# Get ImageNet category names from the weights metadata
categories = weights.meta["categories"]

# Define important labels (in lowercase for easier matching)
important_labels = {"person", "car", "bus", "truck", "traffic light", "stop sign", "bicycle", "motorcycle"}

# ------------------------------
# 3️⃣ Initialize Grad‑CAM Extractor
# ------------------------------
# We'll use SmoothGradCAM++ on the last convolutional layer ("layer4")
cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")

# ------------------------------
# 4️⃣ Define Image Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ResNet50 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------------
# 5️⃣ Set Input & Output Directories
# ------------------------------
# Change this to your CARLA RGB frames directory
INPUT_DIR = "/home/carla1000/InterFuser/interfuser_frame_dataset/RGB"
OUTPUT_DIR = "gradcam_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# 6️⃣ Process a Single Image with Filtering
# ------------------------------
def process_image(image_path):
    """
    Processes an image:
      - Loads and preprocesses the image.
      - Runs the model to obtain a prediction.
      - If the predicted label (from ImageNet) is in the important set,
        it computes Grad-CAM and overlays the heatmap on the original image.
      - Returns the overlay image and the predicted label.
    """
    # Load image and get original size
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (width, height)

    # Preprocess image
    input_tensor = transform(img).unsqueeze(0).to(device)
    # Enable gradient tracking on the input
    input_tensor.requires_grad_(True)

    # Forward pass through the model (we do NOT wrap with torch.no_grad() so gradients are computed)
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    pred_label = categories[pred_class].lower()

    # Filtering: only process images with important labels
    if pred_label not in important_labels:
        print(f"[INFO] Skipping {image_path} as predicted label '{pred_label}' is not in important set.")
        return None

    # Generate Grad-CAM heatmap using torchcam (for the predicted class)
    cam_map = cam_extractor(pred_class, output)[0]
    cam_map = cam_map.squeeze().cpu().detach().numpy()

    # Resize heatmap to original image size
    cam_map = cv2.resize(cam_map, orig_size, interpolation=cv2.INTER_CUBIC)

    # Normalize heatmap to [0, 255]
    cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
    cam_map = np.uint8(255 * cam_map)

    # Convert heatmap to a color map
    heatmap = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)
    return overlay, pred_label

# ------------------------------
# 7️⃣ Process All Images in the Directory
# ------------------------------
def main():
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(image_files)} images in {INPUT_DIR}")

    for img_file in image_files:
        img_path = os.path.join(INPUT_DIR, img_file)
        print(f"[INFO] Processing image: {img_path}")
        result = process_image(img_path)
        if result is None:
            continue
        overlay, pred_label = result
        save_path = os.path.join(OUTPUT_DIR, img_file)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved Grad-CAM overlay: {save_path} (predicted: {pred_label})")

if __name__ == "__main__":
    main()

