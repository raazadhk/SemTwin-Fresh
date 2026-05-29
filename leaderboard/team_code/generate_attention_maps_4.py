#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp
from ultralytics import YOLO  # Import YOLOv8

# ------------------------------
# 1️⃣ Set Device to CPU
# ------------------------------
device = torch.device("cpu")
print(f"[INFO] Using device: {device}")

# ------------------------------
# 2️⃣ Load YOLOv8 for Object Detection
# ------------------------------
print("[INFO] Loading YOLOv8 model on CPU...")
yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8 nano model

# Define important objects for filtering
important_labels = {"person", "car", "bus", "truck", "traffic light", "stop sign", "bicycle", "motorcycle", "trailer truck"}

# ------------------------------
# 3️⃣ Load Pretrained ResNet50 Model for Grad-CAM
# ------------------------------
print("[INFO] Loading pretrained ResNet50 model for Grad-CAM...")
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.to(device)
resnet_model.eval()

# Grad-CAM Extractor
cam_extractor = SmoothGradCAMpp(resnet_model, target_layer="layer4")

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
INPUT_DIRS = {
    "RGB": "/home/carla1000/InterFuser/interfuser_frame_dataset/RGB",
    "RGB_Left": "/home/carla1000/InterFuser/interfuser_frame_dataset/RGB_Left",
    "RGB_Right": "/home/carla1000/InterFuser/interfuser_frame_dataset/RGB_Right"
}
OUTPUT_DIR = "gradcam2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for subdir in INPUT_DIRS.keys():
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

# ------------------------------
# 6️⃣ Process a Single Image
# ------------------------------
def process_image(image_path):
    """
    Detects objects using YOLOv8, filters based on important labels,
    applies Grad-CAM to those detections, and overlays the heatmap.
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (width, height)

    # Run YOLOv8 detection
    results = yolo_model(image_path, verbose=False)[0]

    # Filter detected objects
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls.item())  # Class index
        label = results.names[cls_id]  # Get class name
        confidence = box.conf.item()

        if label in important_labels and confidence > 0.3:  # Confidence threshold
            detections.append((label, confidence, box.xyxy[0].tolist()))

    if not detections:
        print(f"[WARNING] No important objects detected in {image_path}. Skipping...")
        return None

    # Process each detected object
    for label, conf, bbox in detections:
        # Crop object from image
        x1, y1, x2, y2 = map(int, bbox)
        cropped_img = img.crop((x1, y1, x2, y2))

        # Preprocess cropped image for ResNet50
        input_tensor = transform(cropped_img).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        # Forward pass through ResNet50
        output = resnet_model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        # Generate Grad-CAM heatmap
        cam_map = cam_extractor(pred_class, output)[0].squeeze().cpu().detach().numpy()
        cam_map = cv2.resize(cam_map, cropped_img.size, interpolation=cv2.INTER_CUBIC)
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        cam_map = np.uint8(255 * cam_map)
        heatmap = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay Grad-CAM heatmap on cropped image
        overlay = cv2.addWeighted(np.array(cropped_img), 0.6, heatmap, 0.4, 0)

        # Return overlay image
        return overlay, label, conf

# ------------------------------
# 7️⃣ Process All Images in the Directories
# ------------------------------
def main():
    for subdir, input_dir in INPUT_DIRS.items():
        output_subdir = os.path.join(OUTPUT_DIR, subdir)
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not image_files:
            print(f"[ERROR] No images found in {input_dir}")
            continue

        print(f"[INFO] Found {len(image_files)} images in {input_dir}")

        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            print(f"[INFO] Processing image: {img_path}")
            result = process_image(img_path)

            if result is None:
                continue

            overlay, label, conf = result
            save_path = os.path.join(output_subdir, f"{img_file}")
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"[INFO] Saved Grad-CAM: {save_path} (Detected: {label} - {conf:.2f})")

if __name__ == "__main__":
    main()

