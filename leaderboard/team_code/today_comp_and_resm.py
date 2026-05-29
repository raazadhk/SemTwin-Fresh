#!/usr/bin/env python3
import os
import cv2
import numpy as np

# ------------------------------------------
# Configuration
# ------------------------------------------
INPUT_DIR = "/home/carla1000/InterFuser/leaderboard/team_code/gradcam8_results/RGB_Right"
OUTPUT_DIR = "/home/carla1000/q_another_three_by_three_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROWS = 3
COLS = 3

# Threshold for deciding if a tile is "important"
INTENSITY_THRESHOLD = 180

# JPEG Quality Levels
HIGH_QUALITY = 100  # Minimal compression for important tiles (heatmap areas)
LOW_QUALITY = 2    # Heaviest compression for unimportant tiles

# Darkening factor for unimportant tiles
DARKEN_FACTOR = 0.4

# Whether to draw grid lines on the final image
DRAW_GRID = True

# ------------------------------------------
# Check if Tile is Important
# ------------------------------------------
def is_tile_important(tile, threshold):
    """
    Convert the tile to grayscale and check if ANY pixel is above 'threshold'.
    If yes, the tile is considered "important".
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return np.max(gray) > threshold

# ------------------------------------------
# In-Memory JPEG Compression
# ------------------------------------------
def compress_tile(tile, quality):
    """
    Compress 'tile' in memory to the given JPEG 'quality' 
    and return the decompressed (OpenCV) image.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, enc = cv2.imencode('.jpg', tile, encode_param)
    if not success:
        # Fallback: return the original tile if something goes wrong
        return tile
    # Decode back to BGR
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

# ------------------------------------------
# Darken a Tile
# ------------------------------------------
def darken_tile(tile, factor=DARKEN_FACTOR):
    """
    Multiply the tile's pixels by 'factor' to darken them.
    Returns the darkened tile as an 8-bit image.
    """
    darkened = np.clip(tile * factor, 0, 255).astype(np.uint8)
    return darkened

# ------------------------------------------
# Process a Single Image
# ------------------------------------------
def process_image(image_path):
    """
    1) Splits image into 3×3 tiles.
    2) Determines if each tile is important.
    3) If important: compress at HIGH_QUALITY, keep bright.
       If unimportant: compress at LOW_QUALITY, then darken.
    4) Reassemble into one output image with optional grid lines.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    h, w, _ = img.shape
    tile_h = h // ROWS
    tile_w = w // COLS

    # We'll build the final image in this overlay
    overlay = img.copy()

    for row in range(ROWS):
        for col in range(COLS):
            # Calculate tile boundaries
            x_start = col * tile_w
            y_start = row * tile_h
            x_end = (col + 1) * tile_w if col < (COLS - 1) else w
            y_end = (row + 1) * tile_h if row < (ROWS - 1) else h

            tile = overlay[y_start:y_end, x_start:x_end]

            # Determine tile importance
            if is_tile_important(tile, INTENSITY_THRESHOLD):
                # Important tile: minimal compression, keep bright
                tile_processed = compress_tile(tile, HIGH_QUALITY)
            else:
                # Unimportant tile: heavy compression + darken
                tile_compressed = compress_tile(tile, LOW_QUALITY)
                tile_processed = darken_tile(tile_compressed, DARKEN_FACTOR)

            # Put the processed tile back into the final image
            overlay[y_start:y_end, x_start:x_end] = tile_processed

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
    # Get all valid image files
    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Processing {len(image_files)} images from {INPUT_DIR}...")

    for img_file in image_files:
        img_path = os.path.join(INPUT_DIR, img_file)
        output_image = process_image(img_path)
        if output_image is None:
            continue

        # Save the final image
        save_path = os.path.join(OUTPUT_DIR, img_file)
        # Choose a final “container” quality; 95 is typical. 
        # (Tiles are already compressed or not.)
        cv2.imwrite(save_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f" Saved: {save_path}")

if __name__ == "__main__":
    main()

