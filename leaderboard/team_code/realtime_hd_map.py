"""
Real-Time HD Map from Sensor Data
Shows actual roads, lanes, trees, signs, traffic lights from what the agent sees
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


class RealTimeHDMapGenerator:
    """
    Generate HD map showing actual scene from sensor data
    - Roads and lanes from camera + LiDAR
    - Traffic signs and lights from camera detection
    - Trees and poles from LiDAR
    - Vehicles and pedestrians from detections
    """
    
    def __init__(self):
        self.road_color = (60, 60, 60)  # Dark gray
        self.lane_white = (255, 255, 255)
        self.lane_yellow = (0, 255, 255)
        self.tree_color = (34, 139, 34)  # Forest green
        self.pole_color = (169, 169, 169)  # Gray
        
    def generate_hd_map(self,
                       rgb_front: np.ndarray,
                       lidar_data: np.ndarray,
                       traffic_meta: np.ndarray,
                       ego_speed: float,
                       waypoints: np.ndarray = None,
                       map_size: int = 800) -> np.ndarray:
        """
        Generate HD map from sensor data
        
        Args:
            rgb_front: Front camera RGB image
            lidar_data: LiDAR point cloud (Nx3 or Nx4)
            traffic_meta: Traffic detection grid (20x20x7)
            ego_speed: Ego vehicle speed
            waypoints: Predicted waypoints
            map_size: Output map size
            
        Returns:
            HD map image
        """
        
        # Create blank map
        hd_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        hd_map[:] = (20, 20, 20)  # Very dark background
        
        center = map_size // 2
        pixels_per_meter = 10  # 10 pixels per meter
        
        # 1. Extract road structure from LiDAR
        road_points = self._extract_road_from_lidar(lidar_data, pixels_per_meter, center, map_size)
        hd_map = self._draw_road_surface(hd_map, road_points)
        
        # 2. Detect and draw lane markings from camera
        lane_markings = self._detect_lane_markings_from_camera(rgb_front)
        hd_map = self._project_lanes_to_bev(hd_map, lane_markings, pixels_per_meter, center, map_size)
        
        # 3. Extract trees, poles, and static objects from LiDAR
        static_objects = self._extract_static_objects_from_lidar(lidar_data, pixels_per_meter, center, map_size)
        hd_map = self._draw_static_objects(hd_map, static_objects)
        
        # 4. Detect traffic signs and lights from camera
        traffic_elements = self._detect_traffic_elements_from_camera(rgb_front)
        hd_map = self._project_traffic_elements_to_bev(hd_map, traffic_elements, pixels_per_meter, center, map_size)
        
        # 5. Draw detected vehicles and pedestrians
        hd_map = self._draw_dynamic_objects(hd_map, traffic_meta, pixels_per_meter, center, map_size)
        
        # 6. Draw waypoints
        if waypoints is not None:
            hd_map = self._draw_waypoints(hd_map, waypoints, pixels_per_meter, center, map_size)
        
        # 7. Draw ego vehicle
        hd_map = self._draw_ego_vehicle(hd_map, center)
        
        # 8. Add info overlay
        self._add_info_overlay(hd_map, ego_speed)
        
        return hd_map
    
    def _extract_road_from_lidar(self, lidar_data: np.ndarray, 
                                 pixels_per_meter: float, center: int,
                                 map_size: int) -> List[Tuple[int, int]]:
        """Extract road surface points from LiDAR"""
        
        if lidar_data is None or len(lidar_data) == 0:
            return []
        
        # LiDAR points: [x, y, z] or [x, y, z, intensity]
        points = lidar_data[:, :3] if lidar_data.shape[1] >= 3 else lidar_data
        
        # Road is typically low z-values (ground plane)
        ground_threshold = 0.3  # meters above/below sensor
        ground_points = points[np.abs(points[:, 2]) < ground_threshold]
        
        # Convert to BEV coordinates
        road_pixels = []
        for point in ground_points:
            x, y = point[0], point[1]
            
            # Convert to pixels (LiDAR coordinate system)
            px = int(center + y * pixels_per_meter)
            py = int(center - x * pixels_per_meter)
            
            if 0 <= px < map_size and 0 <= py < map_size:
                road_pixels.append((px, py))
        
        return road_pixels
    
    def _draw_road_surface(self, img: np.ndarray, road_points: List[Tuple[int, int]]) -> np.ndarray:
        """Draw road surface from LiDAR points"""
        
        if not road_points:
            return img
        
        # Create road mask
        road_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Draw points
        for px, py in road_points:
            cv2.circle(road_mask, (px, py), 3, 255, -1)
        
        # Dilate to fill road surface
        kernel = np.ones((15, 15), np.uint8)
        road_mask = cv2.dilate(road_mask, kernel, iterations=2)
        
        # Apply road color
        img[road_mask > 0] = self.road_color
        
        return img
    
    def _detect_lane_markings_from_camera(self, rgb_image: np.ndarray) -> Dict:
        """Detect white and yellow lane markings from camera"""
        
        if rgb_image is None or rgb_image.size == 0:
            return {'white_lanes': [], 'yellow_lanes': []}
        
        # Convert to grayscale for white detection
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Detect white markings (bright pixels)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Convert to HSV for yellow detection
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Apply ROI (lower half of image)
        h, w = gray.shape
        roi_mask = np.zeros_like(gray)
        roi_polygon = np.array([[
            (0, h),
            (0, h * 2 // 3),
            (w, h * 2 // 3),
            (w, h)
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_polygon, 255)
        
        white_mask = cv2.bitwise_and(white_mask, roi_mask)
        yellow_mask = cv2.bitwise_and(yellow_mask, roi_mask)
        
        # Find contours
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'white_lanes': white_contours,
            'yellow_lanes': yellow_contours,
            'image_height': h,
            'image_width': w
        }
    
    def _project_lanes_to_bev(self, img: np.ndarray, lane_markings: Dict,
                             pixels_per_meter: float, center: int,
                             map_size: int) -> np.ndarray:
        """Project detected lanes from camera to BEV"""
        
        if not lane_markings:
            return img
        
        img_h = lane_markings.get('image_height', 600)
        img_w = lane_markings.get('image_width', 800)
        
        # Draw white lanes
        for contour in lane_markings.get('white_lanes', []):
            if len(contour) < 10:  # Skip small contours
                continue
            
            points_bev = []
            for point in contour[::5]:  # Subsample
                x, y = point[0]
                
                # Simple perspective projection
                # Bottom of image (high y) = close, top (low y) = far
                depth = (img_h - y) / img_h  # 0 = far, 1 = close
                if depth < 0.3:  # Only use close points
                    continue
                
                # Lateral position
                lateral = (x - img_w / 2) / (img_w / 2)  # -1 to 1
                
                # BEV coordinates
                bev_x = lateral * 20 * depth  # Scale by depth
                bev_y = (1 - depth) * 30  # Forward distance
                
                px = int(center + bev_x * pixels_per_meter)
                py = int(center - bev_y * pixels_per_meter)
                
                if 0 <= px < map_size and 0 <= py < map_size:
                    points_bev.append([px, py])
            
            if len(points_bev) > 2:
                pts = np.array(points_bev, dtype=np.int32)
                cv2.polylines(img, [pts], False, self.lane_white, 2, cv2.LINE_AA)
        
        # Draw yellow lanes (same process)
        for contour in lane_markings.get('yellow_lanes', []):
            if len(contour) < 10:
                continue
            
            points_bev = []
            for point in contour[::5]:
                x, y = point[0]
                depth = (img_h - y) / img_h
                if depth < 0.3:
                    continue
                
                lateral = (x - img_w / 2) / (img_w / 2)
                bev_x = lateral * 20 * depth
                bev_y = (1 - depth) * 30
                
                px = int(center + bev_x * pixels_per_meter)
                py = int(center - bev_y * pixels_per_meter)
                
                if 0 <= px < map_size and 0 <= py < map_size:
                    points_bev.append([px, py])
            
            if len(points_bev) > 2:
                pts = np.array(points_bev, dtype=np.int32)
                cv2.polylines(img, [pts], False, self.lane_yellow, 2, cv2.LINE_AA)
        
        return img
    
    def _extract_static_objects_from_lidar(self, lidar_data: np.ndarray,
                                          pixels_per_meter: float, center: int,
                                          map_size: int) -> Dict:
        """Extract trees, poles, and other static objects from LiDAR"""
        
        if lidar_data is None or len(lidar_data) == 0:
            return {'trees': [], 'poles': []}
        
        points = lidar_data[:, :3] if lidar_data.shape[1] >= 3 else lidar_data
        
        # Trees/vegetation: typically higher z-values, scattered
        tree_threshold_low = 0.5
        tree_threshold_high = 5.0
        tree_points = points[(points[:, 2] > tree_threshold_low) & (points[:, 2] < tree_threshold_high)]
        
        # Poles: tall, thin vertical structures
        pole_threshold = 5.0
        pole_points = points[points[:, 2] > pole_threshold]
        
        # Convert to BEV
        trees = []
        for point in tree_points[::10]:  # Subsample
            x, y = point[0], point[1]
            px = int(center + y * pixels_per_meter)
            py = int(center - x * pixels_per_meter)
            if 0 <= px < map_size and 0 <= py < map_size:
                trees.append((px, py))
        
        poles = []
        for point in pole_points[::5]:
            x, y = point[0], point[1]
            px = int(center + y * pixels_per_meter)
            py = int(center - x * pixels_per_meter)
            if 0 <= px < map_size and 0 <= py < map_size:
                poles.append((px, py))
        
        return {'trees': trees, 'poles': poles}
    
    def _draw_static_objects(self, img: np.ndarray, static_objects: Dict) -> np.ndarray:
        """Draw trees, poles, etc."""
        
        # Draw trees as green circles
        for px, py in static_objects.get('trees', []):
            cv2.circle(img, (px, py), 4, self.tree_color, -1)
        
        # Draw poles as gray circles
        for px, py in static_objects.get('poles', []):
            cv2.circle(img, (px, py), 3, self.pole_color, -1)
        
        return img
    
    def _detect_traffic_elements_from_camera(self, rgb_image: np.ndarray) -> List[Dict]:
        """Detect traffic lights and signs from camera (simplified)"""
        
        if rgb_image is None or rgb_image.size == 0:
            return []
        
        elements = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        h, w = rgb_image.shape[:2]
        
        # Detect red (traffic lights, stop signs)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Detect green (traffic lights)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find red objects
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in red_contours:
            if cv2.contourArea(contour) > 100:  # Filter small noise
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    elements.append({'type': 'red_light', 'x': cx, 'y': cy, 'image_h': h, 'image_w': w})
        
        # Find green objects
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in green_contours:
            if cv2.contourArea(contour) > 100:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    elements.append({'type': 'green_light', 'x': cx, 'y': cy, 'image_h': h, 'image_w': w})
        
        return elements
    
    def _project_traffic_elements_to_bev(self, img: np.ndarray, traffic_elements: List[Dict],
                                        pixels_per_meter: float, center: int,
                                        map_size: int) -> np.ndarray:
        """Project traffic signs/lights to BEV"""
        
        for element in traffic_elements:
            x, y = element['x'], element['y']
            img_h, img_w = element['image_h'], element['image_w']
            
            # Simple projection
            depth = (img_h - y) / img_h
            if depth < 0.2:  # Only use close elements
                continue
            
            lateral = (x - img_w / 2) / (img_w / 2)
            bev_x = lateral * 20 * depth
            bev_y = (1 - depth) * 30
            
            px = int(center + bev_x * pixels_per_meter)
            py = int(center - bev_y * pixels_per_meter)
            
            if 0 <= px < map_size and 0 <= py < map_size:
                if element['type'] == 'red_light':
                    cv2.circle(img, (px, py), 8, (0, 0, 255), -1)
                    cv2.circle(img, (px, py), 9, (255, 255, 255), 1)
                elif element['type'] == 'green_light':
                    cv2.circle(img, (px, py), 8, (0, 255, 0), -1)
                    cv2.circle(img, (px, py), 9, (255, 255, 255), 1)
        
        return img
    
    def _draw_dynamic_objects(self, img: np.ndarray, traffic_meta: np.ndarray,
                             pixels_per_meter: float, center: int,
                             map_size: int) -> np.ndarray:
        """Draw detected vehicles and pedestrians"""
        
        if traffic_meta is None:
            return img
        
        if len(traffic_meta.shape) == 2 and traffic_meta.shape == (400, 7):
            traffic_meta = traffic_meta.reshape(20, 20, 7)
        
        for i in range(20):
            for j in range(20):
                cell = traffic_meta[i, j]
                
                vehicle_prob = cell[4]
                pedestrian_prob = cell[5]
                bike_prob = cell[6]
                
                max_prob = max(vehicle_prob, pedestrian_prob, bike_prob)
                
                if max_prob > 0.4:  # Higher threshold
                    grid_x = (j - 10) * 2.0
                    grid_y = (10 - i) * 2.0
                    
                    obj_x = grid_x + cell[0]
                    obj_y = grid_y + cell[1]
                    
                    px = int(center + obj_x * pixels_per_meter)
                    py = int(center - obj_y * pixels_per_meter)
                    
                    if 0 <= px < map_size and 0 <= py < map_size:
                        if vehicle_prob == max_prob:
                            self._draw_vehicle_box(img, px, py, cell[3], (0, 0, 255), (20, 10))
                        elif pedestrian_prob == max_prob:
                            cv2.circle(img, (px, py), 6, (0, 255, 0), -1)
                        elif bike_prob == max_prob:
                            self._draw_vehicle_box(img, px, py, cell[3], (255, 200, 0), (12, 6))
        
        return img
    
    def _draw_vehicle_box(self, img: np.ndarray, x: int, y: int,
                         yaw: float, color: Tuple[int, int, int],
                         size: Tuple[int, int]):
        """Draw oriented vehicle box"""
        length, width = size
        corners = np.array([
            [-width/2, -length/2],
            [width/2, -length/2],
            [width/2, length/2],
            [-width/2, length/2]
        ])
        
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        rotated = corners @ rotation.T
        rotated[:, 0] += x
        rotated[:, 1] += y
        
        pts = rotated.astype(np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (255, 255, 255), 1)
    
    def _draw_waypoints(self, img: np.ndarray, waypoints: np.ndarray,
                       pixels_per_meter: float, center: int,
                       map_size: int) -> np.ndarray:
        """Draw waypoints"""
        if len(waypoints.shape) == 1:
            waypoints = waypoints.reshape(-1, 2)
        
        for i, wp in enumerate(waypoints):
            px = int(center + wp[0] * pixels_per_meter)
            py = int(center - wp[1] * pixels_per_meter)
            
            if 0 <= px < map_size and 0 <= py < map_size:
                ratio = i / len(waypoints)
                if ratio < 0.5:
                    color = (0, int(255 * (1 - ratio * 2)), int(255 * ratio * 2))
                else:
                    color = (0, int(255 * (1 - (ratio - 0.5) * 2)), 255)
                
                cv2.circle(img, (px, py), 4, color, -1)
        
        return img
    
    def _draw_ego_vehicle(self, img: np.ndarray, center: int):
        """Draw ego vehicle"""
        corners = np.array([
            [center - 6, center - 12],
            [center + 6, center - 12],
            [center + 6, center + 12],
            [center - 6, center + 12]
        ], dtype=np.int32)
        
        cv2.fillPoly(img, [corners], (100, 255, 100))
        cv2.polylines(img, [corners], True, (255, 255, 255), 2)
        cv2.circle(img, (center, center - 10), 4, (255, 255, 255), -1)
        
        return img
    
    def _add_info_overlay(self, img: np.ndarray, ego_speed: float):
        """Add info overlay"""
        cv2.putText(img, f'Speed: {ego_speed*3.6:.1f} km/h', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, 'Real-Time Sensor View', (10, img.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
