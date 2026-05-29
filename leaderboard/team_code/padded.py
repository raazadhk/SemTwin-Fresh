#!/usr/bin/env python3
import os
import cv2
import numpy as np

# ------------------------------------------
# Configuration
# ------------------------------------------
INPUT_DIR = "/home/carla1000/InterFuser/leaderboard/team_code/gradcam8_results/RGB_Right"
OUTPUT_DIR = "/home/carla1000/2nd_another_three_by_three_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROWS = 3
COLS = 3
OUTPUT_SIZE = (400, 300)  # width x height

INTENSITY_THRESHOLD = 180
HIGH_QUALITY = 100
LOW_QUALITY = 2
DARKEN_FACTOR = 0.4
DRAW_GRID = False  # Removed grid lines

# ------------------------------------------
# Utilities
# ------------------------------------------
def is_tile_important(tile, threshold):
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return np.max(gray) > threshold

def compress_tile(tile, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, enc = cv2.imencode('.jpg', tile, encode_param)
    if not success:
        return tile
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def darken_tile(tile, factor=DARKEN_FACTOR):
    return np.clip(tile * factor, 0, 255).astype(np.uint8)

# ------------------------------------------
# Process a Single Image
# ------------------------------------------
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load {image_path}")
        return None

    h, w = img.shape[:2]
    tile_h, tile_w = h // ROWS, w // COLS

    final_img = np.zeros((h, w, 3), dtype=np.uint8)  # black canvas

    for row in range(ROWS):
        for col in range(COLS):
            x1, y1 = col * tile_w, row * tile_h
            x2 = (col + 1) * tile_w if col < COLS - 1 else w
            y2 = (row + 1) * tile_h if row < ROWS - 1 else h

            tile = img[y1:y2, x1:x2]

            if is_tile_important(tile, INTENSITY_THRESHOLD):
                tile = compress_tile(tile, HIGH_QUALITY)
            else:
                tile = np.zeros_like(tile)  # Replace unimportant tile with black padding

            final_img[y1:y2, x1:x2] = tile

    return final_img

# ------------------------------------------
# Main Loop
# ------------------------------------------
def main():
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"🟡 Found {len(image_files)} images in {INPUT_DIR}")

    for fname in image_files:
        input_path = os.path.join(INPUT_DIR, fname)
        output_image = process_image(input_path)
        if output_image is not None:
            output_path = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(output_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"✅ Saved: {output_path}")
        else:
            print(f"⚠️ Skipped: {fname}")

if __name__ == "__main__":
    main()

