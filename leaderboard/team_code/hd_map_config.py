"""
HD Map Generation Configuration
Defines dataset structure, map parameters, and processing settings
"""

import os
from pathlib import Path

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
class DatasetConfig:
    """Configuration for dataset paths and structure"""

    # Base dataset directory (modify this for your system)
    BASE_DIR = os.environ.get("DATASET_DIR", "/new_ssd/interfuser_complete_dataset")

    # Sensor data directories
    SENSORS = {
        "rgb": os.path.join(BASE_DIR, "RGB"),
        "rgb_left": os.path.join(BASE_DIR, "RGB_Left"),
        "rgb_right": os.path.join(BASE_DIR, "RGB_Right"),
        "bev": os.path.join(BASE_DIR, "BEV"),
        "lidar": os.path.join(BASE_DIR, "LiDAR"),
        "measurements": os.path.join(BASE_DIR, "Measurements"),
        "segmentation_front": os.path.join(BASE_DIR, "Segmentation_Front"),
        "segmentation_left": os.path.join(BASE_DIR, "Segmentation_Left"),
        "segmentation_right": os.path.join(BASE_DIR, "Segmentation_Right"),
        "tracking": os.path.join(BASE_DIR, "Tracking"),
    }

    # Output directories
    OUTPUT_DIR = os.path.join(BASE_DIR, "HD_Maps")
    MAP_TILES_DIR = os.path.join(OUTPUT_DIR, "Tiles")
    MAP_OCCUPANCY_DIR = os.path.join(OUTPUT_DIR, "Occupancy")
    MAP_SEMANTIC_DIR = os.path.join(OUTPUT_DIR, "Semantic")
    MAP_VISUAL_DIR = os.path.join(OUTPUT_DIR, "Visual")
    MAP_COMPRESSED_DIR = os.path.join(OUTPUT_DIR, "Compressed")
    MAP_METADATA_DIR = os.path.join(OUTPUT_DIR, "Metadata")


# ============================================================================
# MAP GENERATION PARAMETERS
# ============================================================================
class MapConfig:
    """Configuration for HD map generation"""

    # Map dimensions and resolution
    MAP_SIZE = 512  # 1024x1024 meters (can be larger/smaller)
    RESOLUTION = 0.1  # 10cm per pixel
    PIXELS_PER_METER = int(1 / RESOLUTION)  # 10 pixels per meter

    # Map is centered on vehicle
    # Negative Y = backward (behind vehicle)
    # Positive Y = forward (in front of vehicle)
    MAP_EXTENT_FORWARD = 100  # meters forward
    MAP_EXTENT_BACKWARD = 50   # meters backward
    MAP_EXTENT_LEFT = 50       # meters to left
    MAP_EXTENT_RIGHT = 50      # meters to right

    # LiDAR to occupancy grid
    LIDAR_HEIGHT_MIN = -1.0    # meters (below ego vehicle)
    LIDAR_HEIGHT_MAX = 3.0     # meters (above ego vehicle)
    LIDAR_GRID_RESOLUTION = 0.1  # 10cm cells
    OCCUPANCY_THRESHOLD = 0.5  # confidence threshold for occupancy

    # Semantic segmentation mapping
    # CARLA semantic classes
    SEMANTIC_CLASSES = {
        0: "unlabeled",
        1: "building",
        2: "fence",
        3: "other",
        4: "pedestrian",
        5: "pole",
        6: "road_line",
        7: "road",
        8: "sidewalk",
        9: "vegetation",
        10: "vehicle",
        11: "wall",
        12: "traffic_sign",
        13: "sky",
        14: "ground",
        15: "bridge",
        16: "rail_track",
        17: "guard_rail",
        18: "traffic_light",
        19: "static",
        20: "dynamic",
        21: "water",
        22: "terrain",
    }

    # Important classes for HD map
    IMPORTANT_CLASSES = [7, 8, 10, 4, 12, 18]  # road, sidewalk, vehicle, pedestrian, sign, light

    # Tile configuration
    TILE_SIZE = 128  # 256x256 pixels per tile
    TILE_SIZE_METERS = TILE_SIZE * RESOLUTION  # ~25.6 meters per tile

    # Compression settings
    JPEG_QUALITY = 85 # 0-100, lower = more compressed #85
    WEBP_QUALITY = 80  # 0-100 #80
    PNG_COMPRESSION = 9  # 0-9

    # Map fusion/merging
    FRAME_BUFFER_SIZE = 100  # Keep last N frames for temporal fusion
    OCCUPANCY_DECAY = 0.95  # Decay factor for older observations

    # Transmission settings
    DELTA_THRESHOLD = 0.05  # Only send tile if >5% changed
    PYRAMID_LEVELS = 3  # For hierarchical transmission (full -> 1/2 -> 1/4)


# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================
class ProcessingConfig:
    """Configuration for processing pipeline"""

    # Batch processing
    BATCH_SIZE = 32  # Process frames in batches
    NUM_WORKERS = 4  # Number of parallel workers

    # Frame range
    START_FRAME = 0
    MAX_FRAMES = None  # None = process all available frames

    # Verbosity
    VERBOSE = True
    LOG_INTERVAL = 10  # Log every N frames

    # Quality assurance
    SAVE_INTERMEDIATE = True  # Save intermediate processing results
    VALIDATE_DATA = True  # Validate sensor data integrity


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_output_directories():
    """Create all output directories if they don't exist"""
    dirs = [
        DatasetConfig.OUTPUT_DIR,
        DatasetConfig.MAP_TILES_DIR,
        DatasetConfig.MAP_OCCUPANCY_DIR,
        DatasetConfig.MAP_SEMANTIC_DIR,
        DatasetConfig.MAP_VISUAL_DIR,
        DatasetConfig.MAP_COMPRESSED_DIR,
        DatasetConfig.MAP_METADATA_DIR,
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created/verified: {dir_path}")


def validate_dataset():
    """Validate that dataset directories exist"""
    missing = []
    for sensor_name, sensor_path in DatasetConfig.SENSORS.items():
        if not os.path.exists(sensor_path):
            missing.append(f"{sensor_name} @ {sensor_path}")

    if missing:
        print("⚠ Missing sensor directories:")
        for m in missing:
            print(f"  - {m}")
        return False

    print("✓ All sensor directories found")
    return True


def get_frame_count():
    """Count available frames in RGB directory"""
    rgb_dir = DatasetConfig.SENSORS["rgb"]
    if not os.path.exists(rgb_dir):
        return 0

    frames = [f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))]
    return len(frames)


if __name__ == "__main__":
    print("HD Map Configuration Utility\n")
    print(f"Dataset Base Dir: {DatasetConfig.BASE_DIR}")
    print(f"Map Size: {MapConfig.MAP_SIZE}x{MapConfig.MAP_SIZE} meters @ {MapConfig.RESOLUTION}m resolution")
    print(f"Tile Size: {MapConfig.TILE_SIZE}x{MapConfig.TILE_SIZE} pixels ({MapConfig.TILE_SIZE_METERS:.1f}m)")

    print("\nValidating dataset...")
    if validate_dataset():
        frame_count = get_frame_count()
        print(f"Available frames: {frame_count}")

    print("\nCreating output directories...")
    create_output_directories()
    print("\n✓ Configuration ready!")
