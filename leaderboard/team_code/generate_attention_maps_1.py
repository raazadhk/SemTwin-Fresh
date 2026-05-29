#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

###############################################################################
# 1) CONFIG / MODEL LOADING
###############################################################################

class GlobalConfig:
    model = "interfuser_baseline"
    model_path = "/home/carla1000/InterFuser/leaderboard/team_code/interfuser.pth.tar"

    # Image input sizes from dataset
    front_size = (800, 600)  # (Width, Height)
    side_size = (400, 300)   # (Width, Height)

config = GlobalConfig()

from timm.models import create_model

print("[INFO] Creating the model architecture...")
net = create_model(config.model, pretrained=False)  
print("[INFO] Loading model weights from:", config.model_path)

checkpoint = torch.load(config.model_path, map_location="cuda")
net.load_state_dict(checkpoint["state_dict"])
net.cuda()
net.eval()

###############################################################################
# 2) REGISTER ATTENTION HOOK
###############################################################################

attention_maps = {}

def get_attention_hook(name):
    def hook(module, input, output):
        attn_shape = output.shape  # Debugging
        print(f"[DEBUG] Attention map shape for {name}: {attn_shape}")
        attention_maps[name] = output.detach().cpu().numpy()
    return hook

# Register the correct attention layer
if hasattr(net, 'self_attention'):
    net.self_attention.register_forward_hook(get_attention_hook("self_attention"))
elif hasattr(net, 'attn'):
    net.attn.register_forward_hook(get_attention_hook("attn"))
else:
    print("[WARNING] No known self-attention layer found. Please adapt layer name.")

###############################################################################
# 3) IMAGE TRANSFORMS
###############################################################################

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def create_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

front_transform = create_transform(config.front_size[::-1])
side_transform = create_transform(config.side_size[::-1])

###############################################################################
# 4) SAVE ATTENTION MAP AS A HEATMAP (WITH PROPER RESIZING)
###############################################################################
def save_attention_as_heatmap(attn_map, save_path, target_size):
    """
    Saves the self-attention map as a heatmap image with proper resizing.
    
    attn_map: NumPy array [N, H, W] with values in [0,1].
    save_path: Output file path (.png).
    target_size: Tuple (width, height) matching the input image.
    """
    attn_map = np.mean(attn_map, axis=0)  # Average across attention heads

    # **Ensure correct reshaping** based on token positions
    if attn_map.shape[0] != target_size[1] or attn_map.shape[1] != target_size[0]:
        attn_map = cv2.resize(attn_map, target_size, interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap)

###############################################################################
# 5) PROCESS A SINGLE IMAGE
###############################################################################

def process_single_image(front_path, left_path, right_path, front_transform, side_transform):
    """
    Process images, pass through InterFuser, and save attention maps.
    """
    attention_maps.clear()

    img_pil_front = Image.open(front_path).convert("RGB") if front_path else None
    img_pil_left = Image.open(left_path).convert("RGB") if left_path else None
    img_pil_right = Image.open(right_path).convert("RGB") if right_path else None

    img_tensor_front = front_transform(img_pil_front).unsqueeze(0).cuda() if img_pil_front else None
    img_tensor_left = side_transform(img_pil_left).unsqueeze(0).cuda() if img_pil_left else None
    img_tensor_right = side_transform(img_pil_right).unsqueeze(0).cuda() if img_pil_right else None

    # Create `rgb_center`
    if img_pil_front is not None:
        width, height = img_pil_front.size
        crop_x1, crop_y1 = width // 4, height // 4
        crop_x2, crop_y2 = crop_x1 * 3, crop_y1 * 3
        img_pil_center = img_pil_front.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        img_tensor_center = side_transform(img_pil_center).unsqueeze(0).cuda()
    else:
        img_tensor_center = None

    # Dummy measurements
    fake_measurements = torch.zeros((1, 7)).cuda()
    fake_target_point = torch.zeros((1, 2)).cuda()
    fake_lidar = torch.zeros((1, 3, 400, 400)).cuda()

    model_input = {
        "rgb": img_tensor_front,
        "rgb_left": img_tensor_left,
        "rgb_right": img_tensor_right,
        "rgb_center": img_tensor_center,
        "measurements": fake_measurements,
        "target_point": fake_target_point,
        "lidar": fake_lidar,
    }

    print(f"[DEBUG] Processing: {front_path}, {left_path}, {right_path}")
    print(f"[DEBUG] Model Input Keys: {model_input.keys()}")

    with torch.no_grad():
        net(model_input)

    # Save attention maps with correct dimensions
    os.makedirs("attention_results", exist_ok=True)
    for layer_name, attn_data in attention_maps.items():
        attn_data = attn_data[0]  
        attn_2d = np.mean(attn_data, axis=0)  

        attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min()) if attn_2d.max() > attn_2d.min() else np.zeros_like(attn_2d)

        base_name = os.path.splitext(os.path.basename(front_path))[0] if front_path else "unknown"
        save_path = os.path.join("attention_results", f"{layer_name}_{base_name}.png")

        target_size = config.front_size if front_path else config.side_size
        save_attention_as_heatmap(attn_2d, save_path, target_size)

###############################################################################
# 6) MAIN: LOOP OVER IMAGE DATASET
###############################################################################
def main():
    base_path = "/home/carla1000/InterFuser/interfuser_frame_dataset"
    front_dir = os.path.join(base_path, "RGB")
    left_dir = os.path.join(base_path, "RGB_Left")
    right_dir = os.path.join(base_path, "RGB_Right")

    front_frames = sorted(os.listdir(front_dir))
    left_frames = sorted(os.listdir(left_dir))
    right_frames = sorted(os.listdir(right_dir))

    for i in range(len(front_frames)):
        front_path = os.path.join(front_dir, front_frames[i]) if i < len(front_frames) else None
        left_path = os.path.join(left_dir, left_frames[i]) if i < len(left_frames) else None
        right_path = os.path.join(right_dir, right_frames[i]) if i < len(right_frames) else None

        print(f"[INFO] Processing frame {i}: {front_path}, {left_path}, {right_path}")
        process_single_image(front_path, left_path, right_path, front_transform, side_transform)

if __name__ == "__main__":
    main()

