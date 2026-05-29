"""
High-Quality BEV Semantic Segmentation Map
Matching Tesla FSD / Waymo style visualization

Creates beautiful point-cloud style BEV semantic maps from sensor data
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class HighQualityBEVMapper:
    """
    Generate beautiful BEV semantic segmentation maps
    Style: LiDAR point cloud visualization with semantic colors
    """
    
    # Semantic classes
    CLASSES = {
        0: 'road',
        1: 'lane_marking',
        2: 'sidewalk',
        3: 'vehicle',
        4: 'pedestrian',
        5: 'cyclist',
        6: 'vegetation',
        7: 'building',
        8: 'other'
    }
    
    # Colors matching your reference image (BGR format)
    COLORS = {
        0: (128, 0, 128),      # road - purple/magenta
        1: (0, 165, 255),      # lane marking - orange/yellow
        2: (180, 180, 180),    # sidewalk - light gray
        3: (0, 0, 200),        # vehicle - red
        4: (0, 100, 255),      # pedestrian - orange
        5: (0, 200, 200),      # cyclist - yellow
        6: (34, 139, 34),      # vegetation - green
        7: (100, 100, 100),    # building - dark gray
        8: (50, 50, 50),       # other - very dark gray
    }
    
    def __init__(self, 
                 bev_size=(800, 800),
                 bev_range=50.0,
                 pixels_per_meter=16,
                 point_size=2):
        """
        Args:
            bev_size: Output image size (H, W)
            bev_range: Range in meters
            pixels_per_meter: Resolution (higher = more detail)
            point_size: Size of each point for rendering
        """
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.pixels_per_meter = pixels_per_meter
        self.point_size = point_size
        
        # Background color (dark)
        self.background_color = (40, 40, 40)
        
    def generate_bev_semantic_map(self,
                                  lidar_data: np.ndarray,
                                  rgb_front: Optional[np.ndarray] = None,
                                  traffic_meta: Optional[np.ndarray] = None,
                                  bev_feature: Optional[np.ndarray] = None) -> Dict:
        """
        Generate high-quality BEV semantic map
        
        Args:
            lidar_data: [N, 4] point cloud (x, y, z, intensity)
            rgb_front: [H, W, 3] front camera (optional, for context)
            traffic_meta: [20, 20, 7] detected objects (optional)
            bev_feature: [C, H, W] BEV features (optional)
            
        Returns:
            dict with 'semantic_map', 'rendered', 'point_cloud'
        """
        H, W = self.bev_size
        
        # Initialize with dark background
        rendered = np.ones((H, W, 3), dtype=np.uint8) * np.array(self.background_color, dtype=np.uint8)
        semantic_map = np.ones((H, W), dtype=np.uint8) * 8  # default: other class
        
        # Track points for density/intensity
        point_density = np.zeros((H, W), dtype=np.float32)
        
        # 1. Process LiDAR points
        if lidar_data is not None and len(lidar_data) > 0:
            semantic_map, rendered, point_density = self._process_lidar_points(
                lidar_data, H, W, semantic_map, rendered, point_density
            )
        
        # 2. Add detected objects from traffic_meta
        if traffic_meta is not None:
            semantic_map, rendered = self._add_detected_objects(
                traffic_meta, H, W, semantic_map, rendered
            )
        
        # 3. Enhance with depth-based intensity falloff
        rendered = self._apply_depth_intensity(rendered, point_density, H, W)
        
        # 4. Add grid for reference
        rendered = self._add_reference_grid(rendered)
        
        return {
            'semantic_map': semantic_map,
            'rendered': rendered,
            'point_density': point_density,
            'class_names': self.CLASSES
        }
    
    def _process_lidar_points(self, lidar_data: np.ndarray, H: int, W: int,
                              semantic_map: np.ndarray, rendered: np.ndarray,
                              point_density: np.ndarray) -> Tuple:
        """
        Process LiDAR points and classify them semantically
        """
        for point in lidar_data:
            x, y, z = point[0], point[1], point[2]
            intensity = point[3] if len(point) > 3 else 0.5
            
            # Convert to pixel coordinates
            pixel_x = int(W/2 + y * self.pixels_per_meter)
            pixel_y = int(H - (x + self.bev_range/2) * self.pixels_per_meter)
            
            if not (0 <= pixel_x < W and 0 <= pixel_y < H):
                continue
            
            # Classify based on height and intensity
            class_id = self._classify_point(z, intensity, x, y)
            
            # Update semantic map
            semantic_map[pixel_y, pixel_x] = class_id
            
            # Update point density for rendering
            point_density[pixel_y, pixel_x] += 1
            
            # Render with appropriate color
            color = self.COLORS[class_id]
            
            # Add intensity variation
            intensity_factor = np.clip(intensity, 0.3, 1.0)
            color_adjusted = tuple(int(c * intensity_factor) for c in color)
            
            # Draw point
            cv2.circle(rendered, (pixel_x, pixel_y), self.point_size, color_adjusted, -1)
        
        return semantic_map, rendered, point_density
    
    def _classify_point(self, z: float, intensity: float, x: float, y: float) -> int:
        """
        Classify LiDAR point based on height, intensity, and position
        
        Height-based classification:
        - z < -1.8: road surface
        - -1.8 < z < -1.0: sidewalk (if on sides)
        - z > 1.0: vegetation, buildings
        
        Intensity-based:
        - High intensity + ground level: lane markings
        """
        # Lane markings: high intensity + ground level
        if intensity > 0.8 and -2.0 < z < -1.3:
            return 1  # lane_marking
        
        # Road surface: ground level
        if -2.3 < z < -1.3:
            # Check if on sides (sidewalk) or center (road)
            lateral_dist = abs(y)
            if lateral_dist > 3.0:  # More than 3m from center
                return 2  # sidewalk
            else:
                return 0  # road
        
        # Elevated points
        if z > 0.5:
            # Vegetation (trees, bushes)
            if 0.5 < z < 5.0:
                return 6  # vegetation
            # Buildings (tall structures)
            elif z > 5.0:
                return 7  # building
        
        # Low obstacles on road (could be vehicles)
        if -1.0 < z < 2.0 and -2.0 < y < 2.0:
            return 3  # vehicle (will be refined by traffic_meta)
        
        return 8  # other
    
    def _add_detected_objects(self, traffic_meta: np.ndarray, H: int, W: int,
                              semantic_map: np.ndarray, rendered: np.ndarray) -> Tuple:
        """
        Add detected vehicles, pedestrians, cyclists from traffic_meta
        """
        if traffic_meta.shape != (20, 20, 7):
            traffic_meta = traffic_meta.reshape(20, 20, 7)
        
        for i in range(20):
            for j in range(20):
                meta = traffic_meta[i, j]
                x, y = meta[0], meta[1]
                bbox_x, bbox_y = meta[3], meta[4]
                label = int(meta[6])
                
                # Check if valid detection
                if label == 0 and abs(x) < 0.01 and abs(y) < 0.01:
                    continue
                
                # Map label: 1=vehicle, 2=pedestrian, 3=cyclist
                if label == 1:
                    class_id = 3  # vehicle
                elif label == 2:
                    class_id = 4  # pedestrian
                elif label == 3:
                    class_id = 5  # cyclist
                else:
                    continue
                
                # Convert to BEV pixels
                pixel_x = int(W/2 + y * self.pixels_per_meter)
                pixel_y = int(H - (x + self.bev_range/2) * self.pixels_per_meter)
                
                # Draw filled bounding box
                bbox_w = max(4, int(bbox_y * self.pixels_per_meter))
                bbox_h = max(4, int(bbox_x * self.pixels_per_meter))
                
                y1 = max(0, pixel_y - bbox_h//2)
                y2 = min(H, pixel_y + bbox_h//2)
                x1 = max(0, pixel_x - bbox_w//2)
                x2 = min(W, pixel_x + bbox_w//2)
                
                # Fill region
                semantic_map[y1:y2, x1:x2] = class_id
                
                # Render with bright color
                color = self.COLORS[class_id]
                cv2.rectangle(rendered, (x1, y1), (x2, y2), color, -1)
                
                # Add outline for emphasis
                cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        return semantic_map, rendered
    
    def _apply_depth_intensity(self, rendered: np.ndarray, 
                               point_density: np.ndarray, 
                               H: int, W: int) -> np.ndarray:
        """
        Apply depth-based intensity falloff for more realistic look
        """
        # Create distance map from ego vehicle (bottom center)
        y_coords, x_coords = np.ogrid[0:H, 0:W]
        ego_x, ego_y = W//2, H
        
        # Distance from ego
        dist = np.sqrt((x_coords - ego_x)**2 + (y_coords - ego_y)**2)
        
        # Normalize to 0-1
        max_dist = np.sqrt((W//2)**2 + H**2)
        dist_norm = np.clip(dist / max_dist, 0, 1)
        
        # Apply falloff
        intensity = 1.0 - 0.5 * dist_norm  # Fade to 50% at max distance
        
        # Apply to rendered image
        for c in range(3):
            rendered[:, :, c] = (rendered[:, :, c] * intensity).astype(np.uint8)
        
        return rendered
    
    def _add_reference_grid(self, rendered: np.ndarray, 
                            grid_spacing_meters: float = 10.0) -> np.ndarray:
        """
        Add subtle grid lines for spatial reference
        """
        H, W = rendered.shape[:2]
        grid_color = (60, 60, 60)  # Subtle gray
        
        grid_spacing_pixels = int(grid_spacing_meters * self.pixels_per_meter)
        
        # Horizontal lines
        for i in range(0, H, grid_spacing_pixels):
            cv2.line(rendered, (0, i), (W, i), grid_color, 1)
        
        # Vertical lines
        for j in range(0, W, grid_spacing_pixels):
            cv2.line(rendered, (j, 0), (j, H), grid_color, 1)
        
        # Center line (ego vehicle lane)
        center_x = W // 2
        cv2.line(rendered, (center_x, 0), (center_x, H), (80, 80, 80), 1)
        
        return rendered
    
    def create_visualization_with_overlay(self, result: Dict, 
                                         velocity: float = 0.0,
                                         gps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create final visualization with overlays
        """
        rendered = result['rendered'].copy()
        semantic_map = result['semantic_map']
        H, W = rendered.shape[:2]
        
        # 1. Add ego vehicle marker
        ego_x, ego_y = W//2, int(H * 0.95)
        
        # Draw ego vehicle as filled triangle (pointing up)
        triangle = np.array([
            [ego_x, ego_y - 20],
            [ego_x - 10, ego_y],
            [ego_x + 10, ego_y]
        ], np.int32)
        
        cv2.fillPoly(rendered, [triangle], (0, 255, 255))  # Cyan
        cv2.polylines(rendered, [triangle], True, (255, 255, 255), 2)
        
        # 2. Add heading indicator
        cv2.arrowedLine(rendered, (ego_x, ego_y - 20), (ego_x, ego_y - 35),
                       (255, 255, 255), 2, tipLength=0.3)
        
        # 3. Add info panel
        panel_height = 120
        panel_width = 250
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 20
        
        y_offset = 20
        cv2.putText(panel, 'BEV Semantic Map', (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(panel, f'Speed: {velocity:.1f} m/s', (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y_offset += 20
        
        # Count objects
        num_vehicles = np.sum(semantic_map == 3)
        num_pedestrians = np.sum(semantic_map == 4)
        num_cyclists = np.sum(semantic_map == 5)
        
        cv2.putText(panel, f'Vehicles: {num_vehicles}', (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)
        y_offset += 18
        
        cv2.putText(panel, f'Pedestrians: {num_pedestrians}', (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1)
        y_offset += 18
        
        cv2.putText(panel, f'Cyclists: {num_cyclists}', (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
        
        # Overlay panel with transparency
        x_offset, y_offset = 10, 10
        alpha = 0.7
        roi = rendered[y_offset:y_offset+panel_height, x_offset:x_offset+panel_width]
        blended = cv2.addWeighted(roi, 1-alpha, panel, alpha, 0)
        rendered[y_offset:y_offset+panel_height, x_offset:x_offset+panel_width] = blended
        
        # 4. Add distance scale
        self._add_distance_scale(rendered)
        
        return rendered
    
    def _add_distance_scale(self, rendered: np.ndarray):
        """Add distance scale bar"""
        H, W = rendered.shape[:2]
        
        # Scale bar at bottom right
        scale_length_meters = 10  # 10 meter scale
        scale_length_pixels = int(scale_length_meters * self.pixels_per_meter)
        
        x_start = W - scale_length_pixels - 20
        x_end = W - 20
        y_pos = H - 30
        
        # Draw scale bar
        cv2.line(rendered, (x_start, y_pos), (x_end, y_pos), (255, 255, 255), 2)
        cv2.line(rendered, (x_start, y_pos - 5), (x_start, y_pos + 5), (255, 255, 255), 2)
        cv2.line(rendered, (x_end, y_pos - 5), (x_end, y_pos + 5), (255, 255, 255), 2)
        
        # Label
        cv2.putText(rendered, f'{scale_length_meters}m', (x_start, y_pos - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# Test
if __name__ == "__main__":
    print("Testing High-Quality BEV Mapper...")
    
    # Create sample data that looks realistic
    np.random.seed(42)
    
    # Generate realistic LiDAR data
    n_points = 5000
    lidar_data = []
    
    # Ground points (road)
    for _ in range(3000):
        x = np.random.uniform(0, 40)
        y = np.random.uniform(-6, 6)
        z = np.random.uniform(-2.0, -1.7)
        intensity = np.random.uniform(0.3, 0.7)
        lidar_data.append([x, y, z, intensity])
    
    # Lane markings (high intensity, ground level)
    for x in np.linspace(0, 40, 200):
        for y_offset in [-2.0, 0.0, 2.0]:  # Three lanes
            y = y_offset + np.random.uniform(-0.1, 0.1)
            z = -1.8 + np.random.uniform(-0.1, 0.1)
            intensity = np.random.uniform(0.9, 1.0)
            lidar_data.append([x, y, z, intensity])
    
    # Vegetation on sides
    for _ in range(1000):
        x = np.random.uniform(5, 35)
        y = np.random.choice([-8, -7, 7, 8]) + np.random.uniform(-0.5, 0.5)
        z = np.random.uniform(0.5, 3.0)
        intensity = np.random.uniform(0.2, 0.5)
        lidar_data.append([x, y, z, intensity])
    
    # Objects (vehicles)
    for i in range(3):
        x_base = 15 + i * 8
        y_base = -2.0 + i * 1.5
        for _ in range(100):
            x = x_base + np.random.uniform(-1.2, 1.2)
            y = y_base + np.random.uniform(-0.5, 0.5)
            z = np.random.uniform(-0.5, 1.5)
            intensity = np.random.uniform(0.3, 0.6)
            lidar_data.append([x, y, z, intensity])
    
    lidar_data = np.array(lidar_data)
    
    # Traffic meta with detected vehicles
    traffic_meta = np.zeros((20, 20, 7))
    traffic_meta[12, 9] = [15.0, -2.0, 0.0, 4.0, 1.8, 10.0, 1]  # vehicle
    traffic_meta[14, 10] = [23.0, -0.5, 0.0, 4.0, 1.8, 12.0, 1]  # vehicle
    traffic_meta[16, 11] = [31.0, 1.0, 0.0, 4.0, 1.8, 15.0, 1]   # vehicle
    
    # Generate BEV map
    mapper = HighQualityBEVMapper(bev_size=(800, 800), bev_range=50.0, pixels_per_meter=16)
    
    result = mapper.generate_bev_semantic_map(
        lidar_data=lidar_data,
        traffic_meta=traffic_meta
    )
    
    # Create visualization
    viz = mapper.create_visualization_with_overlay(result, velocity=12.5)
    
    print(f"✓ Semantic map generated: {result['semantic_map'].shape}")
    print(f"✓ Classes found: {np.unique(result['semantic_map'])}")
    print(f"✓ Visualization created: {viz.shape}")
    
    # Save
    cv2.imwrite('/home/claude/high_quality_bev.png', viz)
    cv2.imwrite('/home/claude/high_quality_bev_raw.png', result['rendered'])
    
    print("✓ Outputs saved:")
    print("  - /home/claude/high_quality_bev.png (with overlays)")
    print("  - /home/claude/high_quality_bev_raw.png (raw)")
    
    print("\n✓ High-Quality BEV Mapper ready!")
