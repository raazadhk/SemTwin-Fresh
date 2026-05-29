"""
Enhanced BEV Map Renderer
Makes semantic BEV maps look like professional HD maps
Adds lane detection from camera images
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional


class EnhancedBEVRenderer:
    """
    Enhances BEV semantic maps to look like professional HD maps
    """
    
    def __init__(self):
        self.lane_detector = LaneDetectorFromCamera()
        
    def enhance_bev_map(self, 
                       bev_map: np.ndarray,
                       traffic_meta: np.ndarray,
                       ego_speed: float,
                       waypoints: np.ndarray = None) -> np.ndarray:
        """
        Transform semantic BEV map into HD map style
        
        Args:
            bev_map: Original BEV map (400x400 or similar)
            traffic_meta: Traffic meta grid (20x20x7)
            ego_speed: Ego vehicle speed
            waypoints: Predicted waypoints
            
        Returns:
            Enhanced HD-style map
        """
        
        # Resize to working resolution
        if bev_map.shape[0] != 800:
            bev_map = cv2.resize(bev_map, (800, 800))
        
        # Create HD style map
        hd_map = np.zeros((800, 800, 3), dtype=np.uint8)
        
        # Enhanced background (dark with slight blue tint)
        hd_map[:] = (25, 20, 15)
        
        # Extract road regions from BEV map
        # BEV map usually has road as brighter regions
        gray_bev = cv2.cvtColor(bev_map, cv2.COLOR_BGR2GRAY) if len(bev_map.shape) == 3 else bev_map
        
        # Enhance road surface
        road_mask = gray_bev > 30
        road_regions = cv2.dilate(road_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
        
        # Draw road surface with gradient
        hd_map[road_regions > 0] = (45, 45, 45)
        
        # Add road edges (darker borders)
        edges = cv2.Canny(road_regions * 255, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        hd_map[edges_dilated > 0] = (60, 60, 60)
        
        # Extract and draw lane lines from BEV
        lane_lines = self._extract_lane_lines_from_bev(gray_bev)
        hd_map = self._draw_lane_lines(hd_map, lane_lines)
        
        # Draw grid for reference (subtle)
        hd_map = self._draw_reference_grid(hd_map, pixels_per_meter=20)
        
        # Add detected objects from traffic_meta
        if traffic_meta is not None:
            hd_map = self._draw_detected_objects(hd_map, traffic_meta)
        
        # Draw waypoints if available
        if waypoints is not None:
            hd_map = self._draw_waypoints(hd_map, waypoints, ego_speed)
        
        # Draw ego vehicle
        hd_map = self._draw_ego_vehicle(hd_map)
        
        # Add subtle motion blur effect based on speed
        if ego_speed > 5.0:  # > 18 km/h
            hd_map = self._add_motion_effect(hd_map, ego_speed)
        
        return hd_map
    
    def _extract_lane_lines_from_bev(self, gray_bev: np.ndarray) -> List[np.ndarray]:
        """Extract lane line candidates from BEV map"""
        lines = []
        
        # Use edge detection
        edges = cv2.Canny(gray_bev, 50, 150)
        
        # Detect lines using Hough transform
        detected_lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=40,
            maxLineGap=20
        )
        
        if detected_lines is not None:
            for line in detected_lines:
                x1, y1, x2, y2 = line[0]
                
                # Filter for near-vertical lines (lanes typically go forward)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
                if angle > np.pi/4 and angle < 3*np.pi/4:  # Roughly vertical
                    lines.append(line[0])
        
        return lines
    
    def _draw_lane_lines(self, img: np.ndarray, lines: List[np.ndarray]) -> np.ndarray:
        """Draw lane lines with proper styling"""
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Determine line color based on position
            center_x = img.shape[1] // 2
            line_x = (x1 + x2) // 2
            
            # Lines near center are yellow, sides are white
            if abs(line_x - center_x) < 50:
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 255, 255)  # White
            
            # Draw with anti-aliasing
            cv2.line(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Add glow effect
            cv2.line(img, (x1, y1), (x2, y2), tuple(c//3 for c in color), 4, cv2.LINE_AA)
        
        return img
    
    def _draw_reference_grid(self, img: np.ndarray, pixels_per_meter: float) -> np.ndarray:
        """Draw subtle reference grid"""
        h, w = img.shape[:2]
        
        grid_color = (35, 35, 35)  # Subtle gray
        
        # Vertical lines every 5 meters
        for x in range(0, w, int(pixels_per_meter * 5)):
            cv2.line(img, (x, 0), (x, h), grid_color, 1)
        
        # Horizontal lines every 5 meters
        for y in range(0, h, int(pixels_per_meter * 5)):
            cv2.line(img, (0, y), (w, y), grid_color, 1)
        
        # Highlight center lines
        center_x = w // 2
        center_y = h // 2
        cv2.line(img, (center_x, 0), (center_x, h), (45, 45, 45), 1)
        cv2.line(img, (0, center_y), (w, center_y), (45, 45, 45), 1)
        
        return img
    
    def _draw_detected_objects(self, img: np.ndarray, traffic_meta: np.ndarray) -> np.ndarray:
        """Draw detected objects from traffic_meta grid"""
        
        # traffic_meta is 20x20x7
        # Channels: [dx, dy, speed, orientation, vehicle_prob, pedestrian_prob, bike_prob]
        
        pixels_per_meter = 20  # Assuming 20x20 grid covers 40x40 meters
        center = img.shape[0] // 2
        
        for i in range(20):
            for j in range(20):
                cell = traffic_meta[i, j]
                
                # Check if there's a significant object
                max_prob = np.max(cell[4:7])
                if max_prob > 0.3:
                    # Determine object type
                    obj_type = np.argmax(cell[4:7])
                    obj_types = ['vehicle', 'pedestrian', 'bike']
                    
                    # Calculate position
                    grid_x = (j - 10) * 2.0 * pixels_per_meter
                    grid_y = (10 - i) * 2.0 * pixels_per_meter
                    
                    # Add object offset
                    obj_x = grid_x + cell[0] * pixels_per_meter
                    obj_y = grid_y + cell[1] * pixels_per_meter
                    
                    # Convert to image coordinates
                    px = int(center + obj_x)
                    py = int(center - obj_y)
                    
                    if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                        # Draw based on type
                        if obj_types[obj_type] == 'vehicle':
                            self._draw_vehicle_box(img, px, py, cell[3], (255, 100, 100), (18, 8))
                        elif obj_types[obj_type] == 'pedestrian':
                            self._draw_vehicle_box(img, px, py, 0, (100, 255, 100), (6, 6))
                        elif obj_types[obj_type] == 'bike':
                            self._draw_vehicle_box(img, px, py, cell[3], (100, 200, 255), (12, 6))
                        
                        # Draw velocity vector if moving
                        if cell[2] > 0.5:  # Speed > 0.5 m/s
                            vel_scale = cell[2] * 3
                            vel_x = int(np.cos(cell[3]) * vel_scale * pixels_per_meter)
                            vel_y = int(-np.sin(cell[3]) * vel_scale * pixels_per_meter)
                            
                            cv2.arrowedLine(
                                img, 
                                (px, py), 
                                (px + vel_x, py + vel_y),
                                (255, 255, 0), 
                                2, 
                                tipLength=0.3
                            )
        
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
        
        # Rotate
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        rotated_corners = corners @ rotation_matrix.T
        
        # Translate
        rotated_corners[:, 0] += x
        rotated_corners[:, 1] += y
        
        # Draw with glow effect
        cv2.fillPoly(img, [rotated_corners.astype(int)], tuple(c//2 for c in color))
        cv2.fillPoly(img, [rotated_corners.astype(int)], color)
        cv2.polylines(img, [rotated_corners.astype(int)], True, (255, 255, 255), 1, cv2.LINE_AA)
    
    def _draw_waypoints(self, img: np.ndarray, waypoints: np.ndarray, ego_speed: float) -> np.ndarray:
        """Draw predicted waypoints"""
        
        center = img.shape[0] // 2
        pixels_per_meter = 20
        
        # Reshape waypoints if needed
        if len(waypoints.shape) == 1:
            waypoints = waypoints.reshape(-1, 2)
        
        # Draw waypoints with color gradient (green -> yellow -> red)
        for i, wp in enumerate(waypoints):
            px = int(center + wp[0] * pixels_per_meter)
            py = int(center - wp[1] * pixels_per_meter)
            
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                # Color based on distance (green = close, red = far)
                progress = i / len(waypoints)
                if progress < 0.5:
                    # Green to yellow
                    color = (0, int(255 * (1 - progress * 2)), int(255 * progress * 2))
                else:
                    # Yellow to red
                    color = (0, int(255 * (1 - (progress - 0.5) * 2)), 255)
                
                # Draw waypoint
                cv2.circle(img, (px, py), 4, color, -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Connect waypoints
                if i > 0:
                    prev_wp = waypoints[i-1]
                    prev_px = int(center + prev_wp[0] * pixels_per_meter)
                    prev_py = int(center - prev_wp[1] * pixels_per_meter)
                    cv2.line(img, (prev_px, prev_py), (px, py), color, 2, cv2.LINE_AA)
        
        return img
    
    def _draw_ego_vehicle(self, img: np.ndarray):
        """Draw ego vehicle at center"""
        
        center_x = img.shape[1] // 2
        center_y = img.shape[0] // 2
        
        # Vehicle dimensions
        length = 25
        width = 12
        
        corners = np.array([
            [center_x - width//2, center_y - length//2],
            [center_x + width//2, center_y - length//2],
            [center_x + width//2, center_y + length//2],
            [center_x - width//2, center_y + length//2]
        ], dtype=np.int32)
        
        # Draw with glow
        cv2.fillPoly(img, [corners], (50, 200, 50))
        cv2.fillPoly(img, [corners], (100, 255, 100))
        cv2.polylines(img, [corners], True, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw heading indicator
        front_y = center_y - length//2
        cv2.circle(img, (center_x, front_y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        
        return img
    
    def _add_motion_effect(self, img: np.ndarray, speed: float) -> np.ndarray:
        """Add subtle motion blur based on speed"""
        
        # Normalize speed (max blur at 20 m/s)
        blur_amount = int(min(speed / 20.0 * 3, 3))
        
        if blur_amount > 0:
            # Vertical motion blur (road moving backward)
            kernel = np.zeros((blur_amount * 2 + 1, 1))
            kernel[:, 0] = 1.0 / len(kernel)
            
            # Apply only to bottom half (ground plane)
            bottom_half = img[img.shape[0]//2:, :]
            blurred = cv2.filter2D(bottom_half, -1, kernel)
            img[img.shape[0]//2:, :] = blurred
        
        return img


class LaneDetectorFromCamera:
    """
    Detect lane markings from front camera image
    Projects lanes onto BEV map
    """
    
    def __init__(self):
        self.prev_lanes = None
        
    def detect_lanes(self, rgb_image: np.ndarray) -> Dict:
        """
        Detect lane markings from RGB camera
        
        Args:
            rgb_image: Front camera RGB image
            
        Returns:
            Dictionary with detected lane information
        """
        
        if rgb_image is None or rgb_image.size == 0:
            return {'left_lane': None, 'right_lane': None, 'center_lane': None}
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Detect white and yellow markings
        white_lanes = self._detect_white_markings(rgb_image)
        yellow_lanes = self._detect_yellow_markings(hsv)
        
        # Combine detections
        lane_mask = cv2.bitwise_or(white_lanes, yellow_lanes)
        
        # Apply region of interest (lower half of image)
        roi_mask = np.zeros_like(lane_mask)
        h, w = lane_mask.shape
        roi_polygon = np.array([[
            (0, h),
            (0, h//2),
            (w, h//2),
            (w, h)
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_polygon, 255)
        lane_mask = cv2.bitwise_and(lane_mask, roi_mask)
        
        # Find lane lines
        lanes = self._fit_lane_lines(lane_mask)
        
        # Smooth with previous detection
        if self.prev_lanes is not None:
            lanes = self._smooth_lanes(lanes, self.prev_lanes)
        
        self.prev_lanes = lanes
        
        return lanes
    
    def _detect_white_markings(self, rgb_image: np.ndarray) -> np.ndarray:
        """Detect white lane markings"""
        
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # White markings are bright
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        return white_mask
    
    def _detect_yellow_markings(self, hsv_image: np.ndarray) -> np.ndarray:
        """Detect yellow lane markings"""
        
        # Yellow in HSV: H=20-40, S=100-255, V=100-255
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        
        return yellow_mask
    
    def _fit_lane_lines(self, lane_mask: np.ndarray) -> Dict:
        """Fit polynomial curves to detected lanes"""
        
        h, w = lane_mask.shape
        
        # Find lane pixels using histogram
        histogram = np.sum(lane_mask[h//2:, :], axis=0)
        midpoint = w // 2
        
        # Find left and right peaks
        left_peak = np.argmax(histogram[:midpoint])
        right_peak = np.argmax(histogram[midpoint:]) + midpoint
        
        # Use sliding window to find lane pixels
        left_lane = self._sliding_window(lane_mask, left_peak)
        right_lane = self._sliding_window(lane_mask, right_peak)
        
        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'center_lane': None
        }
    
    def _sliding_window(self, binary_img: np.ndarray, base_x: int, 
                       n_windows: int = 9, margin: int = 50) -> Optional[np.ndarray]:
        """Find lane pixels using sliding window"""
        
        h, w = binary_img.shape
        window_height = h // n_windows
        
        lane_pixels_x = []
        lane_pixels_y = []
        
        current_x = base_x
        
        for window in range(n_windows):
            # Window boundaries
            y_low = h - (window + 1) * window_height
            y_high = h - window * window_height
            x_low = max(0, current_x - margin)
            x_high = min(w, current_x + margin)
            
            # Find pixels in window
            nonzero = binary_img[y_low:y_high, x_low:x_high].nonzero()
            
            if len(nonzero[0]) > 50:  # Minimum pixels to be valid
                nonzero_y = nonzero[0] + y_low
                nonzero_x = nonzero[1] + x_low
                
                lane_pixels_x.extend(nonzero_x)
                lane_pixels_y.extend(nonzero_y)
                
                # Recenter window
                current_x = int(np.mean(nonzero_x))
        
        if len(lane_pixels_x) > 100:  # Enough points to fit
            # Fit polynomial
            coeffs = np.polyfit(lane_pixels_y, lane_pixels_x, 2)
            return coeffs
        
        return None
    
    def _smooth_lanes(self, current: Dict, previous: Dict, alpha: float = 0.7) -> Dict:
        """Smooth lane detection with previous frame"""
        
        smoothed = {}
        
        for key in ['left_lane', 'right_lane', 'center_lane']:
            if current[key] is not None and previous[key] is not None:
                smoothed[key] = alpha * current[key] + (1 - alpha) * previous[key]
            elif current[key] is not None:
                smoothed[key] = current[key]
            else:
                smoothed[key] = previous[key]
        
        return smoothed
    
    def project_lanes_to_bev(self, lanes: Dict, img_height: int) -> np.ndarray:
        """Project detected lanes onto BEV coordinates"""
        
        bev_lanes = []
        
        for lane_type in ['left_lane', 'right_lane']:
            coeffs = lanes.get(lane_type)
            
            if coeffs is not None:
                # Generate points along the lane
                y_points = np.linspace(img_height//2, img_height, 50)
                x_points = np.polyval(coeffs, y_points)
                
                # Convert to BEV coordinates (simplified perspective transform)
                bev_points = []
                for x, y in zip(x_points, y_points):
                    # Perspective projection (approximation)
                    bev_x = (x - img_height//2) * (1.0 + (y - img_height//2) / img_height)
                    bev_y = (img_height - y) * 0.1  # Scale factor
                    
                    bev_points.append([bev_x, bev_y])
                
                bev_lanes.append(np.array(bev_points))
        
        return bev_lanes
