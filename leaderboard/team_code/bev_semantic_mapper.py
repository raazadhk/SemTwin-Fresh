"""
Bird's Eye View Semantic Segmentation HD Map
Clean geometric semantic segmentation using LiDAR point cloud

Semantic Classes:
- Road (Gray)
- Sidewalk (Bright Pink)
- Lane Markings (Yellow)
- Vehicles (Blue)
- Vegetation (Green)
- Buildings (Black)
"""

import cv2
import numpy as np
from typing import Dict, Optional


class BEVSemanticMapper:
    """
    Simple BEV semantic HD map generator
    Uses geometric rules on LiDAR point cloud
    """
    
    # Color scheme matching reference image (BGR format)
    COLORS = {
        'road': (90, 90, 160),          # Purple-Gray
        'sidewalk': (255, 0, 255),      # Bright Pink/Magenta
        'lane': (0, 255, 255),          # Yellow
        'vehicle': (255, 0, 0),         # Blue
        'vegetation': (0, 128, 0),      # Dark Green
        'building': (0, 0, 0),          # Black
        'background': (0, 0, 0)         # Black
    }
    
    CLASS_IDS = {
        'background': 0,
        'road': 1,
        'sidewalk': 2,
        'lane': 3,
        'vehicle': 4,
        'vegetation': 5,
        'building': 6
    }
    
    def __init__(self, 
                 bev_size=(400, 400),
                 bev_range=50.0,
                 pixels_per_meter=8):
        self.bev_size = bev_size
        self.bev_range = bev_range  
        self.pixels_per_meter = pixels_per_meter
        
    def generate(self, lidar_data: np.ndarray, 
                traffic_meta: Optional[np.ndarray] = None) -> Dict:
        """
        Generate BEV semantic map
        
        Args:
            lidar_data: [N, 4] point cloud (x, y, z, intensity)
            traffic_meta: [20, 20, 7] detected objects
            
        Returns:
            {'semantic_map': np.ndarray, 'rendered': np.ndarray}
        """
        H, W = self.bev_size
        
        # Semantic map (class IDs)
        semantic_map = np.zeros((H, W), dtype=np.uint8)
        
        # Process LiDAR points
        if lidar_data is not None and len(lidar_data) > 0:
            semantic_map = self._classify_lidar_points(lidar_data, semantic_map)
        
        # Add detected vehicles
        if traffic_meta is not None:
            semantic_map = self._add_detected_objects(traffic_meta, semantic_map)
        
        # Render to RGB
        rendered = self._render(semantic_map)
        
        return {
            'semantic_map': semantic_map,
            'rendered': rendered
        }
    
    def _classify_lidar_points(self, lidar_data: np.ndarray, 
                               semantic_map: np.ndarray) -> np.ndarray:
        """Classify each LiDAR point geometrically"""
        H, W = semantic_map.shape
        
        for point in lidar_data:
            x, y, z = point[0], point[1], point[2]
            intensity = point[3] if len(point) > 3 else 0.5
            
            # Convert to BEV pixel coordinates
            px = int(W/2 + y * self.pixels_per_meter)
            py = int(H - (x + self.bev_range/2) * self.pixels_per_meter)
            
            if not (0 <= px < W and 0 <= py < H):
                continue
            
            # Geometric classification
            lateral_distance = abs(y)
            
            # Ground level (-2.2m to -1.3m)
            if -2.2 < z < -1.3:
                if intensity > 0.85 and lateral_distance < 3.5:
                    # High intensity on road = lane markings
                    class_id = self.CLASS_IDS['lane']
                elif lateral_distance > 3.5:
                    # Far from center = sidewalk
                    class_id = self.CLASS_IDS['sidewalk']
                else:
                    # Center = road
                    class_id = self.CLASS_IDS['road']
            
            # Elevated vegetation (0.5m to 4m)
            elif 0.5 < z < 4.0:
                if lateral_distance > 3.0:
                    class_id = self.CLASS_IDS['vegetation']
                else:
                    continue
            
            # Tall structures (> 4m)
            elif z > 4.0:
                class_id = self.CLASS_IDS['building']
            
            else:
                continue
            
            # Draw filled circle for smooth appearance
            cv2.circle(semantic_map, (px, py), 2, class_id, -1)
        
        return semantic_map
    
    def _add_detected_objects(self, traffic_meta: np.ndarray,
                              semantic_map: np.ndarray) -> np.ndarray:
        """Add detected vehicles/pedestrians/cyclists"""
        H, W = semantic_map.shape
        
        if traffic_meta.shape != (20, 20, 7):
            traffic_meta = traffic_meta.reshape(20, 20, 7)
        
        for i in range(20):
            for j in range(20):
                meta = traffic_meta[i, j]
                x, y = meta[0], meta[1]
                bbox_x, bbox_y = meta[3], meta[4]
                label = int(meta[6])
                
                # Skip invalid detections
                if label == 0 and abs(x) < 0.01:
                    continue
                
                # All objects shown as vehicles (blue)
                if label in [1, 2, 3]:
                    px = int(W/2 + y * self.pixels_per_meter)
                    py = int(H - (x + self.bev_range/2) * self.pixels_per_meter)
                    
                    # Draw bounding box
                    w = max(3, int(bbox_y * self.pixels_per_meter))
                    h = max(3, int(bbox_x * self.pixels_per_meter))
                    
                    y1 = max(0, py - h//2)
                    y2 = min(H, py + h//2)
                    x1 = max(0, px - w//2)
                    x2 = min(W, px + w//2)
                    
                    cv2.rectangle(semantic_map, (x1, y1), (x2, y2),
                                self.CLASS_IDS['vehicle'], -1)
        
        return semantic_map
    
    def _render(self, semantic_map: np.ndarray) -> np.ndarray:
        """Render semantic map to RGB image"""
        H, W = semantic_map.shape
        rendered = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Map each class ID to its color
        for class_name, class_id in self.CLASS_IDS.items():
            mask = semantic_map == class_id
            rendered[mask] = self.COLORS[class_name]
        
        return rendered
