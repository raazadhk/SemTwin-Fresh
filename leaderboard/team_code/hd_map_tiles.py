"""
HD Map Tile System
Divides full map into tiles for efficient transmission and storage
"""

import numpy as np
import cv2
import os
import json
from typing import Tuple, Dict, List, Optional
from hd_map_config import MapConfig, DatasetConfig


class MapTileGenerator:
    """Generates map tiles from full resolution maps"""

    def __init__(self):
        self.config = MapConfig()
        self.grid_size = int(self.config.MAP_SIZE / self.config.RESOLUTION)
        self.tile_size = self.config.TILE_SIZE

    def divide_into_tiles(
        self,
        full_map: np.ndarray,
        stride: Optional[int] = None
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Divide full map into tiles (default: non-overlapping grid)

        Args:
            full_map: HxW or HxWxC map
            stride: Stride for tile extraction (default: tile_size for non-overlapping)

        Returns:
            tiles: Dictionary {(row, col): tile_data}
        """
        if stride is None:
            stride = self.tile_size

        tiles = {}
        h, w = full_map.shape[:2]

        for i in range(0, h - self.tile_size + 1, stride):
            for j in range(0, w - self.tile_size + 1, stride):
                tile_row = i // self.tile_size
                tile_col = j // self.tile_size

                if len(full_map.shape) == 2:
                    tile = full_map[i:i+self.tile_size, j:j+self.tile_size]
                else:
                    tile = full_map[i:i+self.tile_size, j:j+self.tile_size, :]

                tiles[(tile_row, tile_col)] = tile

        return tiles

    def tiles_to_coordinates(
        self,
        tile_row: int,
        tile_col: int,
        vehicle_position: np.ndarray,
        resolution: float = None
    ) -> Tuple[float, float]:
        """
        Convert tile coordinates to GPS coordinates

        Args:
            tile_row: Tile row index
            tile_col: Tile column index
            vehicle_position: [x, y] ego vehicle position in map
            resolution: meters per pixel (default: config resolution)

        Returns:
            (x, y): GPS coordinates of tile center
        """
        if resolution is None:
            resolution = self.config.RESOLUTION

        grid_center = self.grid_size / 2
        tile_center_pixel_x = (tile_col + 0.5) * self.tile_size
        tile_center_pixel_y = (tile_row + 0.5) * self.tile_size

        # Convert pixels to meters relative to vehicle
        meters_x = (tile_center_pixel_x - grid_center) * resolution
        meters_y = (tile_center_pixel_y - grid_center) * resolution

        # Add vehicle position
        x = vehicle_position[0] + meters_x
        y = vehicle_position[1] + meters_y

        return (x, y)

    def merge_tiles(
        self,
        tiles: Dict[Tuple[int, int], np.ndarray],
        output_shape: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        Merge tiles back into full map

        Args:
            tiles: Dictionary of tiles
            output_shape: (H, W) or (H, W, C) of output map

        Returns:
            full_map: Merged map
        """
        if output_shape is None:
            output_shape = (self.grid_size, self.grid_size)

        if len(output_shape) == 2:
            full_map = np.zeros(output_shape, dtype=tiles[list(tiles.keys())[0]].dtype)
        else:
            full_map = np.zeros(output_shape, dtype=tiles[list(tiles.keys())[0]].dtype)

        for (tile_row, tile_col), tile in tiles.items():
            i = tile_row * self.tile_size
            j = tile_col * self.tile_size

            if len(tile.shape) == 2:
                full_map[i:i+self.tile_size, j:j+self.tile_size] = tile
            else:
                full_map[i:i+self.tile_size, j:j+self.tile_size, :] = tile

        return full_map


class TileMetadata:
    """Manages metadata for tiles"""

    def __init__(self):
        self.config = MapConfig()
        self.tile_generator = MapTileGenerator()

    def create_tile_metadata(
        self,
        tile_row: int,
        tile_col: int,
        frame_id: int,
        timestamp: float,
        vehicle_position: np.ndarray,
        tile_data: np.ndarray,
        compressed_size: int = None,
        uncompressed_size: int = None
    ) -> Dict:
        """
        Create metadata entry for a tile

        Args:
            tile_row, tile_col: Tile indices
            frame_id: Frame number
            timestamp: Timestamp of frame
            vehicle_position: Vehicle GPS position
            tile_data: Tile data (for computing hash)
            compressed_size: Compressed size in bytes
            uncompressed_size: Uncompressed size in bytes

        Returns:
            metadata: Dictionary with tile metadata
        """
        gps_coords = self.tile_generator.tiles_to_coordinates(
            tile_row, tile_col, vehicle_position
        )

        # Compute simple hash of tile content
        tile_hash = hash(tile_data.tobytes()) % (2**32)

        metadata = {
            'tile_id': f"{tile_row:04d}_{tile_col:04d}",
            'tile_row': int(tile_row),
            'tile_col': int(tile_col),
            'frame_id': int(frame_id),
            'timestamp': float(timestamp),
            'gps_x': float(gps_coords[0]),
            'gps_y': float(gps_coords[1]),
            'size_meters': self.config.TILE_SIZE_METERS,
            'hash': int(tile_hash),
            'compressed_size': int(compressed_size) if compressed_size else None,
            'uncompressed_size': int(uncompressed_size) if uncompressed_size else None,
        }

        return metadata

    def save_metadata_index(
        self,
        all_metadata: List[Dict],
        output_path: str
    ):
        """Save metadata index for all tiles"""
        index = {
            'total_tiles': len(all_metadata),
            'tile_size': self.config.TILE_SIZE,
            'tile_size_meters': self.config.TILE_SIZE_METERS,
            'tiles': all_metadata
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(index, f, indent=2)


class PyramidMapGenerator:
    """Generate hierarchical pyramid of maps for efficient transmission"""

    def __init__(self):
        self.config = MapConfig()

    def create_pyramid(
        self,
        full_map: np.ndarray,
        levels: int = 3
    ) -> Dict[int, np.ndarray]:
        """
        Create image pyramid (Laplacian/Gaussian pyramid)

        Args:
            full_map: Full resolution map HxWxC
            levels: Number of pyramid levels

        Returns:
            pyramid: Dictionary {level: map_at_level}
                level 0 = full resolution
                level 1 = 1/2 resolution
                level 2 = 1/4 resolution, etc.
        """
        pyramid = {0: full_map}

        current = full_map
        for level in range(1, levels):
            # Downsample using Gaussian blur then subsampling
            if len(current.shape) == 2:
                downsampled = cv2.pyrDown(current)
            else:
                downsampled = cv2.pyrDown(current)

            pyramid[level] = downsampled
            current = downsampled

        return pyramid

    def pyramid_to_tiles(
        self,
        pyramid: Dict[int, np.ndarray],
        tile_size: int = 256
    ) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Divide pyramid levels into tiles

        Args:
            pyramid: Pyramid from create_pyramid
            tile_size: Size of tiles

        Returns:
            pyramid_tiles: {(level, row, col): tile}
        """
        pyramid_tiles = {}
        tile_gen = MapTileGenerator()

        for level, level_map in pyramid.items():
            h, w = level_map.shape[:2]
            for i in range(0, h - tile_size + 1, tile_size):
                for j in range(0, w - tile_size + 1, tile_size):
                    tile_row = i // tile_size
                    tile_col = j // tile_size

                    if len(level_map.shape) == 2:
                        tile = level_map[i:i+tile_size, j:j+tile_size]
                    else:
                        tile = level_map[i:i+tile_size, j:j+tile_size, :]

                    pyramid_tiles[(level, tile_row, tile_col)] = tile

        return pyramid_tiles


if __name__ == "__main__":
    print("HD Map Tile System")
    print("Available classes:")
    print("  - MapTileGenerator: Divide maps into tiles")
    print("  - TileMetadata: Manage tile metadata and indexing")
    print("  - PyramidMapGenerator: Create hierarchical map pyramid")
