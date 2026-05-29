#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp
from ultralytics import YOLO   # YOLOv8 for object detection

# ──────────────────── 1.  Setup  ────────────────────
device = torch.device("cpu")
print(f"[INFO] Using device: {device}")

yolo_model   = YOLO("yolov8n.pt")
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.to(device).eval()
cam_extractor = SmoothGradCAMpp(resnet_model, target_layer="layer4")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

important_labels = {
    "person", "car", "bus", "truck", "traffic light",
    "stop sign", "bicycle", "motorcycle", "trailer truck"
}

INPUT_DIRS = {
    "RGB":        "/home/carla1000/InterFuser/leaderboard/interfuser_frame_dataset/RGB",
    "RGB_Left":   "/home/carla1000/InterFuser/leaderboard/interfuser_frame_dataset/RGB_Left",
    "RGB_Right":  "/home/carla1000/InterFuser/leaderboard/interfuser_frame_dataset/RGB_Right",
}
OUTPUT_DIR = "gradcam10raj_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
for name in INPUT_DIRS:
    os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)

# ──────────────────── 2.  Worker  ────────────────────
def process_and_save(task):
    """task = (image_path, output_subdir)"""
    image_path, output_subdir = task
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    orig_w, orig_h = img.size

    # YOLO inference
    results = yolo_model(image_path, verbose=False)[0]

    full_heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
    valid = False

    for box in results.boxes:
        cls_id = int(box.cls.item())
        label  = results.names[cls_id]
        conf   = box.conf.item()
        if label in important_labels and conf > 0.3:
            valid = True
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cropped = img.crop((x1, y1, x2, y2))
            tensor  = transform(cropped).unsqueeze(0).to(device)
            tensor.requires_grad_(True)

            out = resnet_model(tensor)
            pred_class = out.argmax(dim=1).item()

            cam = cam_extractor(pred_class, out)[0].squeeze().cpu().numpy()
            cam = cv2.resize(cam, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            full_heatmap[y1:y2, x1:x2] = np.maximum(full_heatmap[y1:y2, x1:x2], cam)

    if not valid:
        print(f"[SKIP] {image_path}")
        return

    full_heatmap = np.uint8(255 * full_heatmap)
    heatmap = cv2.applyColorMap(full_heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    fname = os.path.basename(image_path)
    save_path = os.path.join(output_subdir, fname)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[SAVED] {save_path}")

# ──────────────────── 3.  Build task list & run  ────────────────────
def main():
    tasks = []
    for cam_name, input_dir in INPUT_DIRS.items():
        out_dir = os.path.join(OUTPUT_DIR, cam_name)
        imgs = sorted(f for f in os.listdir(input_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not imgs:
            print(f"[WARN] No images in {input_dir}")
            continue
        tasks.extend([(os.path.join(input_dir, f), out_dir) for f in imgs])

    print(f"[INFO] Processing {len(tasks)} images with {cpu_count()} CPU cores…")
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_and_save, tasks)

if __name__ == "__main__":
    main()

