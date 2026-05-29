#!/usr/bin/env python3
import os
import cv2
import numpy as np

# ------------------------------------------
# Configuration
# ------------------------------------------
INPUT_DIR = "/home/carla1000/InterFuser/leaderboard/team_code/gradcam8_results/RGB_Right"  # Input frames (attention maps)
TILES_DIR = "/home/carla1000/2_compressed_tiles"  # Where tiles are stored
OUTPUT_DIR = "/home/carla1000/2_reassembled_frames"  # Where final images go
os.makedirs(TILES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3×3 Grid
ROWS = 3
COLS = 3

# Threshold to decide if a tile is "important"
INTENSITY_THRESHOLD = 180

# JPEG Quality Levels
NO_COMPRESSION_QUALITY = 100  # Practically no lossy compression for important tiles
HEAVY_COMPRESSION_QUALITY = 10  # Heaviest compression for unimportant tiles

# ------------------------------------------
# Check if Tile is Important
# ------------------------------------------
def is_tile_important(tile, threshold):
    """
    Convert tile to grayscale and check if ANY pixel
    is above 'threshold'. If yes, this tile is "important."
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return np.max(gray) > threshold

# ------------------------------------------
# Split & Compress Each Image
# ------------------------------------------
def split_and_compress_image(image, base_filename):
    """
    1) Splits 'image' into a 3×3 grid.
    2) Determines whether each tile is important or not.
    3) Saves important tiles with NO_COMPRESSION_QUALITY (100), 
       and unimportant tiles with HEAVY_COMPRESSION_QUALITY (10).
    """
    h, w, _ = image.shape
    tile_h = h // ROWS
    tile_w = w // COLS

    for row in range(ROWS):
        for col in range(COLS):
            # Compute tile boundaries
            x_start = col * tile_w
            y_start = row * tile_h
            x_end = (col + 1) * tile_w if col < (COLS - 1) else w
            y_end = (row + 1) * tile_h if row < (ROWS - 1) else h

            tile = image[y_start:y_end, x_start:x_end]

            # Check importance
            if is_tile_important(tile, INTENSITY_THRESHOLD):
                quality = NO_COMPRESSION_QUALITY
            else:
                quality = HEAVY_COMPRESSION_QUALITY

            # Build tile filename
            tile_filename = f"{base_filename}_tile_{row}_{col}.jpg"
            tile_path = os.path.join(TILES_DIR, tile_filename)

            # Write tile with selected quality
            cv2.imwrite(tile_path, tile, [cv2.IMWRITE_JPEG_QUALITY, quality])

# ------------------------------------------
# Reassemble Tiles into Final Frame
# ------------------------------------------
def reassemble_image(base_filename, original_shape):
    """
    Reads back the 3×3 tiles from disk and reconstructs
    them into a single image. Saves the final image to OUTPUT_DIR.
    """
    h, w = original_shape
    tile_h = h // ROWS
    tile_w = w // COLS

    # Create an empty array for reassembled image
    reassembled = np.zeros((h, w, 3), dtype=np.uint8)

    for row in range(ROWS):
        for col in range(COLS):
            tile_filename = f"{base_filename}_tile_{row}_{col}.jpg"
            tile_path = os.path.join(TILES_DIR, tile_filename)
            if not os.path.exists(tile_path):
                print(f"⚠️ Missing tile: {tile_path}")
                continue

            tile = cv2.imread(tile_path)
            # Resize tile to match exactly in case of dimension rounding
            tile = cv2.resize(tile, (tile_w, tile_h))

            y_start = row * tile_h
            x_start = col * tile_w
            reassembled[y_start:y_start+tile_h, x_start:x_start+tile_w] = tile

    # Save the final reassembled image
    output_path = os.path.join(OUTPUT_DIR, f"{base_filename}.jpg")
    cv2.imwrite(output_path, reassembled)
    print(f"✅ Reassembled frame saved: {output_path}")

# ------------------------------------------
# Main Loop
# ------------------------------------------
def main():
    # Get all valid image files from INPUT_DIR
    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if not image_files:
        print(f"❌ No images found in {INPUT_DIR}")
        return

    print(f"📢 Processing {len(image_files)} images from {INPUT_DIR}...")

    for img_file in image_files:
        img_path = os.path.join(INPUT_DIR, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Failed to load {img_path}")
            continue

        # Remove extension for naming
        base_filename, _ = os.path.splitext(img_file)

        # 1) Split & compress each tile
        split_and_compress_image(image, base_filename)

        # 2) Reassemble tiles into the final frame
        reassemble_image(base_filename, image.shape[:2])

if __name__ == "__main__":
    main()

