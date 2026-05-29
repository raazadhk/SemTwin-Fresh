#!/usr/bin/env python3
import os
import cv2
import numpy as np

# ------------------------------------------
# Configuration
# ------------------------------------------
INPUT_DIR = "/home/carla1000/InterFuser/leaderboard/team_code/gradcam2_results/RGB_Right"  
OUTPUT_DIR = "/home/carla1000/b_another_three_by_three_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROWS = 3
COLS = 3

# We check if any pixel in a tile > THRESHOLD
INTENSITY_THRESHOLD = 180  # Adjust to catch smaller hotspots
LIGHTEN_FACTOR = 1.3
DARKEN_FACTOR = 0.5

# Whether to draw the white grid lines
DRAW_GRID = True

# ------------------------------------------
# Lighten or Darken a Tile
# ------------------------------------------
def apply_tile_effect(tile, lighten=True):
    factor = LIGHTEN_FACTOR if lighten else DARKEN_FACTOR
    # Multiply tile by factor and clip
    result = np.clip(tile * factor, 0, 255).astype(np.uint8)
    return result

# ------------------------------------------
# Check if Tile is Important
# ------------------------------------------
def is_tile_important(tile, threshold):
    """
    Convert tile to grayscale and check if 
    ANY pixel is above the threshold. 
    This helps catch small hotspots.
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    max_val = np.max(gray)
    return max_val > threshold

# ------------------------------------------
# Main Processing
# ------------------------------------------
def process_image(image_path):
    """
    1) Splits image into 3x3 tiles
    2) Marks tile as important if any pixel in tile > INTENSITY_THRESHOLD
    3) Lighten important tiles, darken others
    4) Optionally draws white grid lines
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    h, w, _ = img.shape
    tile_h = h // ROWS
    tile_w = w // COLS

    overlay = img.copy()

    # Loop through 3x3 grid
    for row in range(ROWS):
        for col in range(COLS):
            # Calculate tile boundaries
            x_start = col * tile_w
            y_start = row * tile_h

            # Handle edge cases if h or w not divisible by 3
            x_end = (col+1)*tile_w if col < COLS-1 else w
            y_end = (row+1)*tile_h if row < ROWS-1 else h

            tile = overlay[y_start:y_end, x_start:x_end]

            # Check if tile is important
            if is_tile_important(tile, INTENSITY_THRESHOLD):
                # Lighten
                overlay[y_start:y_end, x_start:x_end] = apply_tile_effect(tile, lighten=True)
            else:
                # Darken
                overlay[y_start:y_end, x_start:x_end] = apply_tile_effect(tile, lighten=False)

    # Draw grid lines if desired
    if DRAW_GRID:
        # Horizontal lines
        for r in range(1, ROWS):
            y = r * tile_h
            cv2.line(overlay, (0, y), (w, y), (255,255,255), 2)
        # Vertical lines
        for c in range(1, COLS):
            x = c * tile_w
            cv2.line(overlay, (x, 0), (x, h), (255,255,255), 2)

    return overlay

def main():
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png','.jpg'))])
    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Processing {len(image_files)} images from {INPUT_DIR}...")

    for img_file in image_files:
        img_path = os.path.join(INPUT_DIR, img_file)
        result = process_image(img_path)
        if result is None:
            continue

        save_path = os.path.join(OUTPUT_DIR, img_file)
        cv2.imwrite(save_path, result)
        print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    main()

