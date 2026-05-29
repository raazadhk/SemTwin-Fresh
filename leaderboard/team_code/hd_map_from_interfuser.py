"""
Zero-Training HD Map Generation from InterFuser Outputs

Uses your EXISTING InterFuser predictions:
- traffic_meta (20x20x7) - vehicles, pedestrians, bikes with positions
- bev_feature - for road structure (optional)
- No training needed!
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class HDMapFromInterfuser:
    """
    Generate HD maps directly from InterFuser's existing outputs
    NO TRAINING REQUIRED!
    
    Uses:
    - traffic_meta: [20, 20, 7] - your existing vehicle/pedestrian predictions
    - (Optional) Simple heuristics for road geometry
    """
    
    # Color scheme (BGR for OpenCV)
    COLORS = {
        'background': (50, 50, 50),
        'drivable': (128, 128, 128),
        'lane_marking': (255, 255, 255),
        'road_edge': (0, 255, 255),
        'vehicle': (0, 255, 0),
        'pedestrian': (255, 0, 0),
        'cyclist': (0, 255, 255),
    }
    
    def __init__(self, bev_size=(200, 200), pixels_per_meter=10):
        """
        Args:
            bev_size: Output HD map size
            pixels_per_meter: Resolution (10 means 20m x 20m for 200x200 map)
        """
        self.bev_size = bev_size
        self.pixels_per_meter = pixels_per_meter
        
    def generate_hd_map(self, 
                       traffic_meta: np.ndarray,
                       velocity: float = 0.0,
                       create_road_geometry: bool = True) -> Dict:
        """
        Generate HD map from InterFuser's traffic_meta
        
        Args:
            traffic_meta: [400, 7] or [20, 20, 7] - your existing predictions
                         Format: [x, y, yaw, bbox_x, bbox_y, speed, label]
                         label: 1=vehicle, 2=pedestrian, 3=cyclist
            velocity: ego vehicle speed (for road geometry estimation)
            create_road_geometry: whether to add road/lane markings
            
        Returns:
            dict with 'static_map', 'dynamic_map', 'rendered_map'
        """
        H, W = self.bev_size
        
        # Reshape if needed
        if traffic_meta.shape == (400, 7):
            traffic_meta = traffic_meta.reshape(20, 20, 7)
        
        # Create static map (road geometry)
        if create_road_geometry:
            static_map = self._create_road_geometry(H, W, velocity)
        else:
            static_map = np.zeros((H, W), dtype=np.uint8)
        
        # Create dynamic map from traffic_meta
        dynamic_map = self._create_dynamic_map_from_traffic_meta(
            traffic_meta, H, W
        )
        
        # Render to RGB
        rendered_map = self._render_maps(static_map, dynamic_map)
        
        return {
            'static_map': static_map,
            'dynamic_map': dynamic_map,
            'rendered_map': rendered_map,
            'traffic_meta': traffic_meta
        }
    
    def _create_road_geometry(self, H: int, W: int, velocity: float) -> np.ndarray:
        """
        Create simple road geometry using heuristics
        (No ML needed - just geometric rules)
        """
        static_map = np.zeros((H, W), dtype=np.uint8)
        
        # Define road area (assume ego is at bottom center)
        road_width_pixels = int(W * 0.4)  # Road is 40% of width
        center = W // 2
        
        # Drivable area (class 2)
        left_edge = center - road_width_pixels // 2
        right_edge = center + road_width_pixels // 2
        static_map[:, left_edge:right_edge] = 2
        
        # Lane markings (class 1) - center line
        lane_center = center
        static_map[:, lane_center-1:lane_center+1] = 1
        
        # Road edges (class 3)
        static_map[:, left_edge-2:left_edge] = 3
        static_map[:, right_edge:right_edge+2] = 3
        
        # Optional: Add curve based on velocity (dynamic road)
        # if velocity > 5.0:  # Only at higher speeds
        #     Add curvature estimation
        
        return static_map
    
    def _create_dynamic_map_from_traffic_meta(self, 
                                              traffic_meta: np.ndarray,
                                              H: int, W: int) -> np.ndarray:
        """
        Convert traffic_meta to dynamic semantic map
        This is the KEY function - uses your existing predictions!
        """
        dynamic_map = np.zeros((H, W), dtype=np.uint8)
        
        # traffic_meta shape: [20, 20, 7]
        # Format: [x, y, yaw, bbox_x, bbox_y, speed, label]
        # label: 1=vehicle, 2=pedestrian, 3=cyclist
        
        grid_h, grid_w = traffic_meta.shape[:2]
        
        for i in range(grid_h):
            for j in range(grid_w):
                meta = traffic_meta[i, j]
                
                x, y = meta[0], meta[1]
                bbox_x, bbox_y = meta[3], meta[4]
                label = int(meta[6])
                
                # Check if detection exists (non-zero position or label)
                if label > 0 or (abs(x) > 0.01 or abs(y) > 0.01):
                    
                    # Convert from meters to pixels
                    # Your traffic_meta is in ego-centric coordinates
                    # Assuming: x forward, y lateral, ego at (H, W/2)
                    
                    pixel_x = int(W/2 + y * self.pixels_per_meter)
                    pixel_y = int(H - x * self.pixels_per_meter)
                    
                    # Bounding box size in pixels
                    bbox_w = max(3, int(bbox_y * self.pixels_per_meter))
                    bbox_h = max(3, int(bbox_x * self.pixels_per_meter))
                    
                    # Draw bounding box
                    y1 = max(0, pixel_y - bbox_h//2)
                    y2 = min(H, pixel_y + bbox_h//2)
                    x1 = max(0, pixel_x - bbox_w//2)
                    x2 = min(W, pixel_x + bbox_w//2)
                    
                    # Map label to our classes
                    # InterFuser: 1=vehicle, 2=pedestrian, 3=cyclist
                    # Our classes: 0=vehicle, 1=pedestrian, 2=cyclist
                    if label == 1:
                        class_id = 0  # vehicle
                    elif label == 2:
                        class_id = 1  # pedestrian
                    elif label == 3:
                        class_id = 2  # cyclist
                    else:
                        continue
                    
                    dynamic_map[y1:y2, x1:x2] = class_id
        
        return dynamic_map
    
    def _render_maps(self, static_map: np.ndarray, dynamic_map: np.ndarray) -> np.ndarray:
        """Render semantic maps to RGB"""
        H, W = static_map.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Render static elements
        rgb[static_map == 0] = self.COLORS['background']
        rgb[static_map == 1] = self.COLORS['lane_marking']
        rgb[static_map == 2] = self.COLORS['drivable']
        rgb[static_map == 3] = self.COLORS['road_edge']
        
        # Overlay dynamic elements
        alpha = 0.8
        dynamic_overlay = np.zeros_like(rgb)
        dynamic_overlay[dynamic_map == 0] = self.COLORS['vehicle']
        dynamic_overlay[dynamic_map == 1] = self.COLORS['pedestrian']
        dynamic_overlay[dynamic_map == 2] = self.COLORS['cyclist']
        
        mask = (dynamic_map > 0).astype(np.float32)[:, :, None]
        rgb = (rgb * (1 - alpha * mask) + dynamic_overlay * alpha * mask).astype(np.uint8)
        
        # Add grid
        self._add_grid(rgb, spacing=20)
        
        return rgb
    
    def _add_grid(self, img: np.ndarray, spacing: int = 20, color=(80, 80, 80)):
        """Add grid lines for spatial reference"""
        H, W = img.shape[:2]
        
        for i in range(0, H, spacing):
            cv2.line(img, (0, i), (W, i), color, 1)
        for j in range(0, W, spacing):
            cv2.line(img, (j, 0), (j, H), color, 1)
    
    def create_enhanced_visualization(self,
                                     hd_map_data: Dict,
                                     pred_waypoints: Optional[np.ndarray] = None,
                                     velocity: float = 0.0) -> np.ndarray:
        """
        Create enhanced HD map with ego vehicle, waypoints, and statistics
        
        Args:
            hd_map_data: Output from generate_hd_map()
            pred_waypoints: [N, 2] predicted waypoints in meters
            velocity: current velocity
        """
        rgb = hd_map_data['rendered_map'].copy()
        H, W = rgb.shape[:2]
        
        # Draw ego vehicle at bottom center
        ego_x, ego_y = W//2, H - 20
        self._draw_ego_vehicle(rgb, ego_x, ego_y)
        
        # Draw waypoints if provided
        if pred_waypoints is not None:
            self._draw_waypoints(rgb, pred_waypoints, ego_x, ego_y)
        
        # Add statistics overlay
        self._add_statistics_overlay(rgb, hd_map_data, velocity)
        
        return rgb
    
    def _draw_ego_vehicle(self, img: np.ndarray, x: int, y: int):
        """Draw ego vehicle as yellow arrow"""
        # Vehicle body
        pts = np.array([
            [x-5, y-10], [x+5, y-10],
            [x+5, y+10], [x-5, y+10]
        ], np.int32)
        cv2.fillPoly(img, [pts], (0, 255, 255))  # Yellow
        cv2.polylines(img, [pts], True, (0, 0, 0), 2)
        
        # Direction arrow
        cv2.arrowedLine(img, (x, y), (x, y-15), (255, 255, 255), 2, tipLength=0.3)
    
    def _draw_waypoints(self, img: np.ndarray, waypoints: np.ndarray, 
                       ego_x: int, ego_y: int):
        """Draw predicted trajectory"""
        H, W = img.shape[:2]
        
        for i, wp in enumerate(waypoints):
            # Convert waypoint from meters to pixels
            wp_x = int(ego_x + wp[1] * self.pixels_per_meter)  # lateral
            wp_y = int(ego_y - wp[0] * self.pixels_per_meter)  # forward
            
            if 0 <= wp_x < W and 0 <= wp_y < H:
                color = (0, 255, 0) if i < 5 else (0, 165, 255)
                cv2.circle(img, (wp_x, wp_y), 3, color, -1)
                
                # Connect waypoints
                if i > 0:
                    prev_wp = waypoints[i-1]
                    prev_x = int(ego_x + prev_wp[1] * self.pixels_per_meter)
                    prev_y = int(ego_y - prev_wp[0] * self.pixels_per_meter)
                    if 0 <= prev_x < W and 0 <= prev_y < H:
                        cv2.line(img, (prev_x, prev_y), (wp_x, wp_y), color, 2)
    
    def _add_statistics_overlay(self, img: np.ndarray, hd_map_data: Dict, velocity: float):
        """Add text overlay with statistics"""
        dynamic_map = hd_map_data['dynamic_map']
        
        # Count objects
        num_vehicles = np.sum(dynamic_map == 0)
        num_pedestrians = np.sum(dynamic_map == 1)
        num_cyclists = np.sum(dynamic_map == 2)
        
        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (200, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Statistics
        y_offset = 25
        cv2.putText(img, f"Speed: {velocity:.1f} m/s", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(img, f"Vehicles: {num_vehicles}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20
        cv2.putText(img, f"Pedestrians: {num_pedestrians}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        y_offset += 20
        cv2.putText(img, f"Cyclists: {num_cyclists}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


# ============================================================================
# Simple Integration Example for InterfuserAgent
# ============================================================================

"""
INTEGRATION (No training needed!):

1. In your InterfuserAgent.setup():

    from team_code.hd_map_from_interfuser import HDMapFromInterfuser
    
    self.hd_map_generator = HDMapFromInterfuser(
        bev_size=(200, 200),
        pixels_per_meter=10
    )


2. In your InterfuserAgent.run_step(), after getting traffic_meta:

    # You already have this:
    traffic_meta = self.traffic_meta_moving_avg  # [400, 7]
    
    # Generate HD map (no training needed!)
    hd_map_data = self.hd_map_generator.generate_hd_map(
        traffic_meta=traffic_meta,
        velocity=velocity,
        create_road_geometry=True
    )
    
    # Create enhanced visualization with waypoints
    hd_map_viz = self.hd_map_generator.create_enhanced_visualization(
        hd_map_data,
        pred_waypoints=pred_waypoints,
        velocity=velocity
    )
    
    # Add to tick_data for display
    tick_data['hd_map'] = hd_map_viz
    tick_data['hd_map_raw'] = hd_map_data


3. In DisplayInterface.run_interface():

    if 'hd_map' in input_data:
        hd_map = cv2.resize(input_data['hd_map'], (400, 400))
        surface[0:400, 800:1200] = hd_map
        
        cv2.putText(surface, 'HD Map (Real-time)', (820, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


That's it! No training, no model weights, just use your existing predictions!
"""


if __name__ == "__main__":
    print("=" * 70)
    print("Zero-Training HD Map Generator - Test")
    print("=" * 70)
    
    # Create sample traffic_meta (simulating InterFuser output)
    traffic_meta = np.zeros((20, 20, 7))
    
    # Add some sample vehicles
    traffic_meta[5, 10] = [3.0, -1.0, 0.0, 2.5, 1.0, 5.0, 1]  # Vehicle ahead-left
    traffic_meta[8, 10] = [5.0, 0.0, 0.0, 2.5, 1.0, 8.0, 1]   # Vehicle ahead
    traffic_meta[6, 12] = [4.0, 1.0, 0.0, 2.5, 1.0, 6.0, 1]   # Vehicle ahead-right
    
    # Add pedestrian
    traffic_meta[3, 9] = [2.0, -0.5, 0.0, 0.5, 0.5, 1.0, 2]   # Pedestrian
    
    # Add cyclist
    traffic_meta[7, 11] = [4.5, 0.5, 0.0, 1.0, 0.8, 4.0, 3]   # Cyclist
    
    # Generate HD map
    generator = HDMapFromInterfuser(bev_size=(200, 200), pixels_per_meter=10)
    
    hd_map_data = generator.generate_hd_map(
        traffic_meta=traffic_meta,
        velocity=10.0,
        create_road_geometry=True
    )
    
    # Create sample waypoints
    pred_waypoints = np.array([
        [1.0, 0.0], [2.0, 0.1], [3.0, 0.2],
        [4.0, 0.3], [5.0, 0.4], [6.0, 0.5],
        [7.0, 0.6], [8.0, 0.7], [9.0, 0.8], [10.0, 1.0]
    ])
    
    # Create enhanced visualization
    enhanced_viz = generator.create_enhanced_visualization(
        hd_map_data,
        pred_waypoints=pred_waypoints,
        velocity=10.0
    )
    
    # Save outputs
    cv2.imwrite('/home/claude/hd_map_zero_training.png', enhanced_viz)
    
    print("\n✓ HD Map generated successfully!")
    print(f"  - Static map shape: {hd_map_data['static_map'].shape}")
    print(f"  - Dynamic map shape: {hd_map_data['dynamic_map'].shape}")
    print(f"  - Rendered map shape: {hd_map_data['rendered_map'].shape}")
    print(f"  - Detected vehicles: {np.sum(hd_map_data['dynamic_map'] == 0)}")
    print(f"  - Detected pedestrians: {np.sum(hd_map_data['dynamic_map'] == 1)}")
    print(f"  - Detected cyclists: {np.sum(hd_map_data['dynamic_map'] == 2)}")
    print("\n✓ Output saved to: /home/claude/hd_map_zero_training.png")
    print("\n" + "=" * 70)
    print("NO TRAINING NEEDED - Just integrate and run!")
    print("=" * 70)
