#!/usr/bin/env python3
import os
import cv2
import numpy as np

# ------------------------------------------
# Configuration
# ------------------------------------------
INPUT_DIR = "/home/carla1000/InterFuser/leaderboard/team_code/gradcam8_results/RGB_Right"
OUTPUT_DIR = "/home/carla1000/c_another_three_by_three_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROWS = 3
COLS = 3

# Threshold for deciding if a tile is "important"
INTENSITY_THRESHOLD = 180

# JPEG Qualities
HIGH_QUALITY = 100  # practically no (or minimal) JPEG compression
LOW_QUALITY = 10    # very heavy compression

# Whether to draw the white grid lines
DRAW_GRID = True

# ------------------------------------------
# Check if Tile is Important
# ------------------------------------------
def is_tile_important(tile, threshold):
    """
    Convert tile to grayscale and check if ANY pixel 
    is above the threshold. If yes, we consider 
    this tile 'important'.
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    max_val = np.max(gray)
    return max_val > threshold

# ------------------------------------------
# Compress Tile In-Memory
# ------------------------------------------
def compress_tile(tile, quality):
    """
    Compresses 'tile' to a given JPEG 'quality' in-memory,
    then decodes it back to an OpenCV image. This way, we can
    directly place the compressed version into the final image.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # Encode to JPEG
    result, enc = cv2.imencode('.jpg', tile, encode_param)
    if not result:
        # Fallback: just return original tile if something fails
        return tile

    # Decode back to a normal BGR image
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

# ------------------------------------------
# Process a Single Image
# ------------------------------------------
def process_image(image_path):
    """
    1) Splits image into 3x3 tiles
    2) Determines if a tile is important (contains pixels > INTENSITY_THRESHOLD)
    3) Leaves important tiles uncompressed (HIGH_QUALITY); compresses unimportant tiles (LOW_QUALITY)
    4) Reassembles these tiles into a single output image
    5) Optionally draws white grid lines
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    h, w, _ = img.shape
    tile_h = h // ROWS
    tile_w = w // COLS

    # We'll build the final image in an "overlay" copy
    overlay = img.copy()

    # Loop through 3x3 grid
    for row in range(ROWS):
        for col in range(COLS):
            # Calculate tile boundaries
            x_start = col * tile_w
            y_start = row * tile_h
            x_end = (col + 1) * tile_w if col < COLS - 1 else w
            y_end = (row + 1) * tile_h if row < ROWS - 1 else h

            tile = overlay[y_start:y_end, x_start:x_end]

            # Check if the tile is important
            if is_tile_important(tile, INTENSITY_THRESHOLD):
                # Keep tile at HIGH_QUALITY (practically no compression)
                tile_compressed = compress_tile(tile, HIGH_QUALITY)
            else:
                # Heavily compress tile
                tile_compressed = compress_tile(tile, LOW_QUALITY)

            # Place the (possibly compressed) tile back
            overlay[y_start:y_end, x_start:x_end] = tile_compressed

    # Draw grid lines if desired
    if DRAW_GRID:
        # Horizontal lines
        for r in range(1, ROWS):
            y = r * tile_h
            cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 2)
        # Vertical lines
        for c in range(1, COLS):
            x = c * tile_w
            cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 2)

    return overlay

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
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
        # Finally save the entire image (with compressed/uncompressed tiles)
        # at normal or high quality. If you want to control final output 
        # compression, you can do so here too.
        cv2.imwrite(save_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    main()

