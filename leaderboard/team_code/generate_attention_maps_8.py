#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp
from ultralytics import YOLO  # YOLOv8 for object detection

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
    "RGB": "/home/carla1000/InterFuser/leaderboard/interfuser_frame_dataset/RGB",
    "RGB_Left": "/home/carla1000/InterFuser/leaderboard/interfuser_frame_dataset/RGB_Left",
    "RGB_Right": "/home/carla1000/InterFuser/leaderboard/interfuser_frame_dataset/RGB_Right"
}
OUTPUT_DIR = "gradcam8_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for subdir in INPUT_DIRS.keys():
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

# ------------------------------
# 6️⃣ Process a Single Image
# ------------------------------
def process_image(image_path):
    """
    Detects objects using YOLOv8, filters based on important labels,
    applies Grad-CAM, and overlays it on the full image.
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (width, height)
    img_np = np.array(img)  # Convert to NumPy array

    # Run YOLOv8 detection
    results = yolo_model(image_path, verbose=False)[0]

    # Create a blank heatmap (same size as input image)
    full_heatmap = np.zeros((orig_size[1], orig_size[0]), dtype=np.float32)

    # Process detected objects
    has_valid_detection = False
    for box in results.boxes:
        cls_id = int(box.cls.item())  # Class index
        label = results.names[cls_id]  # Get class name
        confidence = box.conf.item()

        if label in important_labels and confidence > 0.3:  # Confidence threshold
            has_valid_detection = True

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Crop object from image
            cropped_img = img.crop((x1, y1, x2, y2))

            # Preprocess cropped image for ResNet50
            input_tensor = transform(cropped_img).unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)

            # Forward pass through ResNet50
            output = resnet_model(input_tensor)
            pred_class = output.argmax(dim=1).item()

            # Generate Grad-CAM heatmap
            cam_map = cam_extractor(pred_class, output)[0].squeeze().cpu().detach().numpy()
            cam_map = cv2.resize(cam_map, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

            # Normalize heatmap to [0, 1]
            cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)

            # Place heatmap back on full-size frame
            full_heatmap[y1:y2, x1:x2] = np.maximum(full_heatmap[y1:y2, x1:x2], cam_map)

    # Skip images without valid detections
    if not has_valid_detection:
        print(f"[WARNING] No important objects detected in {image_path}. Skipping...")
        return None

    # Normalize final full-frame heatmap
    full_heatmap = np.uint8(255 * full_heatmap)
    heatmap = cv2.applyColorMap(full_heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay Grad-CAM heatmap onto full image
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    return overlay

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
            overlay = process_image(img_path)

            if overlay is None:
                continue

            save_path = os.path.join(output_subdir, img_file)
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"[INFO] Saved Grad-CAM overlay: {save_path}")

if __name__ == "__main__":
    main()

