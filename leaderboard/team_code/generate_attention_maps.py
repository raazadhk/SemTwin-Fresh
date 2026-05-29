#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp

# ------------------------------
# 1️⃣ Set Device to CPU
# ------------------------------
device = torch.device("cpu")
print(f"[INFO] Using device: {device}")

# ------------------------------
# 2️⃣ Load Pretrained ResNet50 Model
# ------------------------------
print("[INFO] Loading pretrained ResNet50 model...")
# Note: Using the weights enum is recommended in newer versions,
# but here we simply use pretrained=True.
model = models.resnet50(pretrained=True)
model.to(device)
model.eval()

# ------------------------------
# 3️⃣ Initialize Grad‑CAM Extractor
# ------------------------------
# We'll use SmoothGradCAM++ on the last convolutional layer, "layer4"
cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")

# ------------------------------
# 4️⃣ Define Image Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------------
# 5️⃣ Set Input & Output Directories
# ------------------------------
INPUT_DIR = "/home/carla1000/InterFuser/interfuser_frame_dataset/RGB"
OUTPUT_DIR = "gradcam_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# 6️⃣ Process a Single Image & Generate Grad‑CAM Overlay
# ------------------------------
def process_image(image_path):
    """
    Processes an image: applies the model, computes Grad‑CAM,
    and returns an overlay image with the heatmap.
    """
    # Load image using PIL and get original size
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (width, height)

    # Preprocess image; ensure gradients are enabled
    input_tensor = transform(img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)  # Enable gradient computation

    # Forward pass (do NOT wrap with torch.no_grad() so that gradients are tracked)
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # Generate Grad‑CAM heatmap using torchcam
    cam_map = cam_extractor(pred_class, output)[0]

    # Convert the heatmap to a NumPy array and resize to original image size
    cam_map = cam_map.squeeze().cpu().detach().numpy()
    cam_map = cv2.resize(cam_map, orig_size, interpolation=cv2.INTER_CUBIC)

    # Normalize the heatmap to the range [0, 255]
    cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
    cam_map = np.uint8(255 * cam_map)

    # Convert heatmap to a color map
    heatmap = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay the heatmap on the original image
    overlay = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)
    return overlay

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
        overlay = process_image(img_path)

        # Save overlay image (convert RGB to BGR for cv2.imwrite)
        save_path = os.path.join(OUTPUT_DIR, img_file)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved Grad-CAM overlay: {save_path}")

if __name__ == "__main__":
    main()

