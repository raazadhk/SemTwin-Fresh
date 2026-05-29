"""
HD Map Layer Generators
Converts sensor data to 2D map layers:
- Occupancy layer from LiDAR
- Semantic layer from segmentation
- Visual layer from RGB cameras
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import json
from hd_map_config import MapConfig, DatasetConfig


class OccupancyGridGenerator:
    """Converts LiDAR point clouds to 2D occupancy grids"""

    def __init__(self):
        self.config = MapConfig()
        self.grid_size = int(self.config.MAP_SIZE / self.config.RESOLUTION)

    def lidar_to_occupancy(
        self,
        lidar_points: np.ndarray,
        vehicle_position: np.ndarray,
        vehicle_rotation: float,
    ) -> np.ndarray:
        """
        Convert LiDAR point cloud to 2D occupancy grid

        Args:
            lidar_points: Nx3 array of 3D points (x, y, z)
            vehicle_position: [x, y] GPS position
            vehicle_rotation: Yaw angle in radians

        Returns:
            grid: HxW occupancy grid (0-255, 0=empty, 255=occupied)
        """
        # Initialize occupancy grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if lidar_points.size == 0:
            return grid

        # Filter points by height (remove ground and too-high points)
        height_mask = (
            (lidar_points[:, 2] >= self.config.LIDAR_HEIGHT_MIN)
            & (lidar_points[:, 2] <= self.config.LIDAR_HEIGHT_MAX)
        )
        lidar_points = lidar_points[height_mask]

        if lidar_points.size == 0:
            return grid

        # Transform points to ego vehicle frame
        # Rotate points
        cos_rot = np.cos(vehicle_rotation)
        sin_rot = np.sin(vehicle_rotation)
        rotation_matrix = np.array([
            [cos_rot, -sin_rot],
            [sin_rot, cos_rot]
        ])

        xy = lidar_points[:, :2]
        xy_rotated = xy @ rotation_matrix.T
        xy_rotated -= vehicle_position

        # Convert to grid coordinates
        # Ego vehicle is at center of grid
        grid_center = self.grid_size / 2
        grid_coords = (xy_rotated / self.config.RESOLUTION + grid_center).astype(np.int32)

        # Filter points within grid bounds
        valid_mask = (
            (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < self.grid_size) &
            (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < self.grid_size)
        )
        grid_coords = grid_coords[valid_mask]

        # Mark occupied cells
        if grid_coords.shape[0] > 0:
            grid[grid_coords[:, 1], grid_coords[:, 0]] = 255

        return grid

    def apply_occupancy_decay(
        self,
        current_grid: np.ndarray,
        previous_grid: np.ndarray,
        decay_factor: float = 0.95
    ) -> np.ndarray:
        """
        Apply temporal decay to occupancy grid for temporal fusion

        Args:
            current_grid: New occupancy observation
            previous_grid: Previous merged grid
            decay_factor: How much to weight previous observations

        Returns:
            merged_grid: Temporally fused grid
        """
        # Normalize to 0-1 range
        current = current_grid.astype(np.float32) / 255.0
        previous = previous_grid.astype(np.float32) / 255.0

        # Max pooling: take the max probability
        merged = np.maximum(
            current,
            previous * decay_factor
        )

        return (merged * 255).astype(np.uint8)


class SemanticLayerGenerator:
    """Creates semantic segmentation layers from multi-view segmentation data"""

    def __init__(self):
        self.config = MapConfig()
        self.grid_size = int(self.config.MAP_SIZE / self.config.RESOLUTION)

    def project_segmentation_to_bev(
        self,
        seg_image: np.ndarray,
        camera_matrix: np.ndarray,
        vehicle_position: np.ndarray,
        vehicle_rotation: float,
        camera_pose: Dict  # {'x': float, 'y': float, 'z': float, 'pitch': float, 'yaw': float, 'roll': float}
    ) -> np.ndarray:
        """
        Project semantic segmentation image to BEV grid

        Args:
            seg_image: HxWx3 semantic segmentation image
            camera_matrix: 3x3 camera intrinsic matrix
            vehicle_position: [x, y] GPS position
            vehicle_rotation: Vehicle yaw angle
            camera_pose: Camera extrinsics relative to vehicle

        Returns:
            bev_grid: HxWx23 grid with one-hot semantic channels
        """
        # Initialize BEV grid (23 semantic classes)
        bev_grid = np.zeros(
            (self.grid_size, self.grid_size, 23),
            dtype=np.uint8
        )

        # Extract semantic class IDs (stored in blue channel)
        semantic_ids = seg_image[:, :, 2]

        # For each pixel in image, project to BEV
        # This is simplified - in real scenario you'd use proper camera to BEV projection
        # with depth estimation or assumed ground plane

        # Simple approach: assume points are on ground plane at z=0
        h, w = semantic_ids.shape
        for class_id in self.config.IMPORTANT_CLASSES:
            class_mask = (semantic_ids == class_id)
            bev_grid[class_mask.astype(bool), class_id] = 255

        return bev_grid

    def merge_segmentation_views(
        self,
        seg_front: np.ndarray,
        seg_left: np.ndarray,
        seg_right: np.ndarray
    ) -> np.ndarray:
        """
        Merge segmentation from multiple camera views

        Args:
            seg_front: Front camera segmentation
            seg_left: Left camera segmentation
            seg_right: Right camera segmentation

        Returns:
            merged_semantic_grid: Combined semantic grid
        """
        grid = np.zeros(
            (self.grid_size, self.grid_size, 23),
            dtype=np.uint8
        )

        # Project each view (simplified)
        # In real implementation, use proper 3D→2D projection

        # Extract important classes from front view
        semantic_ids = seg_front[:, :, 2]
        for class_id in self.config.IMPORTANT_CLASSES:
            class_mask = (semantic_ids == class_id)
            grid[class_mask.astype(bool), class_id] = 255

        return grid


class VisualLayerGenerator:
    """Creates visual texture layers from RGB cameras"""

    def __init__(self):
        self.config = MapConfig()
        self.grid_size = int(self.config.MAP_SIZE / self.config.RESOLUTION)

    def project_rgb_to_bev(
        self,
        rgb_image: np.ndarray,
        depth_estimate: np.ndarray,
        camera_matrix: np.ndarray,
        vehicle_position: np.ndarray,
        vehicle_rotation: float,
    ) -> np.ndarray:
        """
        Project RGB image to BEV using depth estimation

        Args:
            rgb_image: HxWx3 RGB image
            depth_estimate: HxW depth map (estimated from stereo or monocular)
            camera_matrix: 3x3 camera intrinsic matrix
            vehicle_position: [x, y] GPS position
            vehicle_rotation: Vehicle yaw angle

        Returns:
            bev_visual: HxWx3 BEV visual texture
        """
        # Initialize BEV visual layer
        bev_visual = np.zeros(
            (self.grid_size, self.grid_size, 3),
            dtype=np.uint8
        )

        # Simplified: project center-view pixels to BEV
        # Real implementation would use proper perspective transform

        h, w = rgb_image.shape[:2]
        cy, cx = h // 2, w // 2

        # Sample from center region
        region = rgb_image[cy-64:cy+64, cx-64:cx+64]
        center_grid = self.grid_size // 2

        # Place in center of BEV grid
        size = min(region.shape[0], self.grid_size // 4)
        bev_visual[
            center_grid - size//2:center_grid + size//2,
            center_grid - size//2:center_grid + size//2
        ] = cv2.resize(region, (size, size))

        return bev_visual

    def use_bev_camera(
        self,
        bev_image: np.ndarray,
        vehicle_position: np.ndarray,
        vehicle_rotation: float
    ) -> np.ndarray:
        """
        Use BEV camera image directly as visual texture

        Args:
            bev_image: HxWx3 BEV camera image (already bird's eye view)
            vehicle_position: [x, y] GPS position
            vehicle_rotation: Vehicle yaw angle (for rotation correction)

        Returns:
            bev_visual: HxWx3 aligned BEV visual texture
        """
        # Rotate to align with vehicle frame
        h, w = bev_image.shape[:2]
        center = (w // 2, h // 2)

        # Rotate by negative vehicle yaw to align with map frame
        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(vehicle_rotation), 1.0)
        bev_rotated = cv2.warpAffine(
            bev_image,
            rotation_matrix,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        # Resize to grid size if needed
        if bev_rotated.shape[:2] != (self.grid_size, self.grid_size):
            bev_rotated = cv2.resize(
                bev_rotated,
                (self.grid_size, self.grid_size),
                interpolation=cv2.INTER_LINEAR
            )

        return bev_rotated


class LayerFusion:
    """Fuse multiple layers into coherent HD map"""

    def __init__(self):
        self.config = MapConfig()
        self.grid_size = int(self.config.MAP_SIZE / self.config.RESOLUTION)

    def fuse_layers(
        self,
        occupancy: np.ndarray,
        semantic: np.ndarray,
        visual: np.ndarray,
        weights: Dict = None
    ) -> Dict[str, np.ndarray]:
        """
        Fuse occupancy, semantic, and visual layers

        Args:
            occupancy: HxW occupancy grid
            semantic: HxWx23 semantic grid
            visual: HxWx3 visual texture
            weights: Fusion weights for each layer

        Returns:
            fused_map: Dictionary with fused layers
        """
        if weights is None:
            weights = {
                'occupancy': 0.5,
                'semantic': 0.3,
                'visual': 0.2
            }

        # Normalize occupancy to 0-1
        occ_norm = occupancy.astype(np.float32) / 255.0

        # Create fused layer: combine occupancy and semantic info
        fused_occ_sem = np.zeros(
            (self.grid_size, self.grid_size, 24),
            dtype=np.float32
        )

        # Channel 0: occupancy
        fused_occ_sem[:, :, 0] = occ_norm * weights['occupancy']

        # Channels 1-23: semantic channels (normalized)
        semantic_norm = semantic.astype(np.float32) / 255.0
        fused_occ_sem[:, :, 1:24] = semantic_norm * weights['semantic']

        # Visual texture can be stored separately or as additional channels
        return {
            'occupancy': occupancy,
            'semantic': semantic,
            'visual': visual,
            'fused_occ_sem': (fused_occ_sem * 255).astype(np.uint8),
        }

    def save_fused_map(self, fused_map: Dict, frame_id: int, output_dir: str):
        """Save fused map layers"""
        os.makedirs(output_dir, exist_ok=True)

        # Save occupancy
        occ_path = os.path.join(output_dir, f"occ_{frame_id:06d}.png")
        cv2.imwrite(occ_path, fused_map['occupancy'])

        # Save visual
        vis_path = os.path.join(output_dir, f"vis_{frame_id:06d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(fused_map['visual'], cv2.COLOR_RGB2BGR))

        # Save semantic (as NPY for efficiency)
        sem_path = os.path.join(output_dir, f"sem_{frame_id:06d}.npy")
        np.save(sem_path, fused_map['semantic'])


if __name__ == "__main__":
    print("HD Map Layer Generators")
    print("Available classes:")
    print("  - OccupancyGridGenerator: LiDAR → 2D occupancy")
    print("  - SemanticLayerGenerator: Segmentation → BEV semantic")
    print("  - VisualLayerGenerator: RGB → BEV visual texture")
    print("  - LayerFusion: Fuse all layers into coherent map")
