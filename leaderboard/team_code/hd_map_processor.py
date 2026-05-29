"""
HD Map Post-Processor
Main pipeline to convert collected CARLA sensor data to 2D HD maps
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import deque
import time

from hd_map_config import DatasetConfig, MapConfig, ProcessingConfig, create_output_directories, validate_dataset, get_frame_count
from hd_map_layers import OccupancyGridGenerator, SemanticLayerGenerator, VisualLayerGenerator, LayerFusion
from hd_map_tiles import MapTileGenerator, TileMetadata, PyramidMapGenerator
from hd_map_compression import CompressionCodec, DeltaEncoder, AdaptiveCompression


class HDMapProcessor:
    """Main processor for HD map generation"""

    def __init__(self, config: Optional[Dict] = None):
        self.dataset_config = DatasetConfig()
        self.map_config = MapConfig()
        self.proc_config = ProcessingConfig()

        # Merge custom config if provided
        if config:
            for key, value in config.items():
                setattr(self.map_config, key, value)

        # Initialize generators
        self.occ_gen = OccupancyGridGenerator()
        self.sem_gen = SemanticLayerGenerator()
        self.vis_gen = VisualLayerGenerator()
        self.fusion = LayerFusion()
        self.tile_gen = MapTileGenerator()
        self.tile_metadata = TileMetadata()
        self.pyramid_gen = PyramidMapGenerator()
        self.codec = CompressionCodec()
        self.delta_encoder = DeltaEncoder()

        # Frame buffer for temporal fusion
        self.frame_buffer = deque(maxlen=self.map_config.FRAME_BUFFER_SIZE)
        self.merged_occupancy = None

    def load_frame_data(self, frame_id: int) -> Optional[Dict]:
        """
        Load all sensor data for a single frame

        Args:
            frame_id: Frame number (0-indexed)

        Returns:
            frame_data: Dictionary with all sensor data for this frame
        """
        frame_str = f"{frame_id:06d}"
        frame_data = {}

        try:
            # Load RGB images
            rgb_path = os.path.join(self.dataset_config.SENSORS['rgb'], f"{frame_str}.jpg")
            if os.path.exists(rgb_path):
                frame_data['rgb'] = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

            # Load BEV camera
            bev_path = os.path.join(self.dataset_config.SENSORS['bev'], f"{frame_str}.jpg")
            if os.path.exists(bev_path):
                frame_data['bev'] = cv2.cvtColor(cv2.imread(bev_path), cv2.COLOR_BGR2RGB)

            # Load semantic segmentation
            seg_front_path = os.path.join(self.dataset_config.SENSORS['segmentation_front'], f"{frame_str}.png")
            if os.path.exists(seg_front_path):
                frame_data['seg_front'] = cv2.imread(seg_front_path)

            # Load LiDAR
            lidar_path = os.path.join(self.dataset_config.SENSORS['lidar'], f"{frame_str}.npy")
            if os.path.exists(lidar_path):
                frame_data['lidar'] = np.load(lidar_path)

            # Load measurements (GPS, compass, speed, etc.)
            measurements_path = os.path.join(self.dataset_config.SENSORS['measurements'], f"{frame_str}.json")
            if os.path.exists(measurements_path):
                with open(measurements_path, 'r') as f:
                    frame_data['measurements'] = json.load(f)

            # Load tracking data
            tracking_path = os.path.join(self.dataset_config.SENSORS['tracking'], f"{frame_str}.json")
            if os.path.exists(tracking_path):
                with open(tracking_path, 'r') as f:
                    frame_data['tracking'] = json.load(f)

            return frame_data if frame_data else None

        except Exception as e:
            print(f"Error loading frame {frame_id}: {e}")
            return None

    def process_frame(self, frame_id: int, frame_data: Dict) -> Optional[Dict]:
        """
        Process single frame to generate map layers

        Args:
            frame_id: Frame number
            frame_data: Loaded frame data

        Returns:
            map_data: Dictionary with occupancy, semantic, visual, and fused layers
        """
        try:
            map_data = {'frame_id': frame_id}

            # Extract measurements
            measurements = frame_data.get('measurements', {})
            vehicle_pos = np.array([measurements.get('gps', [0, 0])[0], measurements.get('gps', [0, 0])[1]])
            vehicle_rotation = measurements.get('compass', 0)  # radians

            # ========== OCCUPANCY GRID ==========
            if 'lidar' in frame_data:
                lidar = frame_data['lidar']
                occ_grid = self.occ_gen.lidar_to_occupancy(lidar, vehicle_pos, vehicle_rotation)

                # Temporal fusion with previous frames
                if self.merged_occupancy is None:
                    self.merged_occupancy = occ_grid
                else:
                    self.merged_occupancy = self.occ_gen.apply_occupancy_decay(
                        occ_grid,
                        self.merged_occupancy,
                        decay_factor=self.map_config.OCCUPANCY_DECAY
                    )

                map_data['occupancy'] = self.merged_occupancy.copy()

            # ========== SEMANTIC LAYER ==========
            if 'seg_front' in frame_data:
                # For now, use a simplified semantic layer generation
                # In production, use proper camera→BEV projection with depth estimation
                sem_layer = np.zeros(
                    (self.occ_gen.grid_size, self.occ_gen.grid_size, 23),
                    dtype=np.uint8
                )
                map_data['semantic'] = sem_layer

            # ========== VISUAL LAYER ==========
            if 'bev' in frame_data:
                visual_layer = self.vis_gen.use_bev_camera(
                    frame_data['bev'],
                    vehicle_pos,
                    vehicle_rotation
                )
                map_data['visual'] = visual_layer

            # ========== FUSE ALL LAYERS ==========
            if 'occupancy' in map_data and 'visual' in map_data:
                fused = self.fusion.fuse_layers(
                    map_data.get('occupancy'),
                    map_data.get('semantic', np.zeros((self.occ_gen.grid_size, self.occ_gen.grid_size, 23), dtype=np.uint8)),
                    map_data.get('visual')
                )
                map_data.update(fused)

            return map_data

        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")
            return None

    def generate_tiles(
        self,
        map_data: Dict,
        frame_id: int,
        vehicle_pos: np.ndarray
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Generate tiles from processed map

        Args:
            map_data: Processed map data
            frame_id: Frame number
            vehicle_pos: Vehicle position

        Returns:
            tiles_with_metadata: {(row, col): {'data': tile, 'metadata': metadata}}
        """
        tiles_with_metadata = {}

        if 'occupancy' in map_data:
            occ_tiles = self.tile_gen.divide_into_tiles(map_data['occupancy'])

            for (tile_row, tile_col), tile_data in occ_tiles.items():
                tile_key = (tile_row, tile_col)

                # Check if tile changed significantly
                should_update, change_pct = self.delta_encoder.should_update_tile(tile_key, tile_data)

                if should_update:
                    # Compress tile
                    compressed = self.codec.compress_png(tile_data)
                    original_size = tile_data.nbytes

                    # Create metadata
                    metadata = self.tile_metadata.create_tile_metadata(
                        tile_row, tile_col, frame_id,
                        time.time(),
                        vehicle_pos,
                        tile_data,
                        compressed_size=len(compressed),
                        uncompressed_size=original_size
                    )
                    metadata['change_percentage'] = float(change_pct)

                    tiles_with_metadata[tile_key] = {
                        'data': tile_data,
                        'compressed': compressed,
                        'metadata': metadata
                    }

        return tiles_with_metadata

    def save_map_data(self, map_data: Dict, frame_id: int):
        """Save processed map data to disk"""
        frame_str = f"{frame_id:06d}"

        # Save occupancy grid
        if 'occupancy' in map_data:
            occ_path = os.path.join(self.dataset_config.OUTPUT_DIR, "Occupancy", f"occ_{frame_str}.png")
            os.makedirs(os.path.dirname(occ_path), exist_ok=True)
            cv2.imwrite(occ_path, map_data['occupancy'])

        # Save visual layer
        if 'visual' in map_data:
            vis_path = os.path.join(self.dataset_config.OUTPUT_DIR, "Visual", f"vis_{frame_str}.jpg")
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            cv2.imwrite(vis_path, cv2.cvtColor(map_data['visual'], cv2.COLOR_RGB2BGR))

    def process_dataset(self, start_frame: int = 0, max_frames: Optional[int] = None):
        """
        Process entire dataset

        Args:
            start_frame: Starting frame number
            max_frames: Maximum frames to process (None = all)
        """
        print(f"\n{'='*60}")
        print(f"HD Map Processing Pipeline")
        print(f"{'='*60}")

        # Validate dataset
        if not validate_dataset():
            print("Dataset validation failed!")
            return

        # Create output directories
        create_output_directories()

        # Get frame count
        total_frames = get_frame_count()
        if max_frames:
            total_frames = min(total_frames, start_frame + max_frames)

        print(f"Total frames to process: {total_frames - start_frame}")
        print(f"Map config: {self.map_config.MAP_SIZE}x{self.map_config.MAP_SIZE}m @ {self.map_config.RESOLUTION}m resolution")
        print(f"Tile size: {self.map_config.TILE_SIZE}x{self.map_config.TILE_SIZE} pixels\n")

        all_tiles_metadata = []
        frame_count = 0

        for frame_id in range(start_frame, total_frames):
            # Load frame
            frame_data = self.load_frame_data(frame_id)
            if frame_data is None:
                continue

            # Process frame
            map_data = self.process_frame(frame_id, frame_data)
            if map_data is None:
                continue

            # Get vehicle position
            measurements = frame_data.get('measurements', {})
            vehicle_pos = np.array([measurements.get('gps', [0, 0])[0], measurements.get('gps', [0, 0])[1]])

            # Generate tiles
            tiles = self.generate_tiles(map_data, frame_id, vehicle_pos)

            # Collect metadata
            for tile_key, tile_info in tiles.items():
                all_tiles_metadata.append(tile_info['metadata'])

            # Save map data
            self.save_map_data(map_data, frame_id)

            frame_count += 1

            # Log progress
            if frame_count % self.proc_config.LOG_INTERVAL == 0:
                print(f"Processed {frame_count} frames | Generated {len(all_tiles_metadata)} tiles")

        # Save metadata index
        if all_tiles_metadata:
            metadata_path = os.path.join(self.dataset_config.MAP_METADATA_DIR, "tiles_index.json")
            self.tile_metadata.save_metadata_index(all_tiles_metadata, metadata_path)
            print(f"\n✓ Saved metadata index: {metadata_path}")

        print(f"\n{'='*60}")
        print(f"Processing Complete!")
        print(f"{'='*60}")
        print(f"Frames processed: {frame_count}")
        print(f"Tiles generated: {len(all_tiles_metadata)}")
        print(f"Output directory: {self.dataset_config.OUTPUT_DIR}")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="HD Map Post-Processor for CARLA Dataset")
    parser.add_argument("--dataset-dir", default="/new_ssd/interfuser_complete_dataset",
                        help="Path to dataset directory")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Starting frame number")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process")
    parser.add_argument("--map-size", type=int, default=1024,
                        help="Map size in meters")
    parser.add_argument("--resolution", type=float, default=0.1,
                        help="Map resolution in meters/pixel")

    args = parser.parse_args()

    # Override dataset directory if provided
    os.environ["DATASET_DIR"] = args.dataset_dir

    # Create custom config
    custom_config = {
        'MAP_SIZE': args.map_size,
        'RESOLUTION': args.resolution,
    }

    # Create processor
    processor = HDMapProcessor(config=custom_config)

    # Process dataset
    processor.process_dataset(
        start_frame=args.start_frame,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
