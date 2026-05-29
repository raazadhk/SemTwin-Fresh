"""
Vectorized BEV HD Map Generator
Creates clean, Tesla FSD-style HD maps from InterfuserAgent sensors
- Extracts road geometry
- Vectorizes lane boundaries
- Shows lane markings as clean lines
- Processes LiDAR into structured map
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


class VectorizedHDMap:
    """
    Generate clean vectorized HD maps from sensor data
    No messy point clouds - clean geometric representations
    """
    
    def __init__(self):
        # Colors
        self.bg_color = (25, 25, 25)  # Dark background
        self.road_color = (50, 50, 50)  # Road surface
        self.lane_boundary_color = (50, 100, 255)  # Orange-red
        self.lane_marking_color = (255, 255, 255)  # White
        self.lane_center_color = (0, 255, 255)  # Yellow
        self.ego_color = (200, 200, 200)  # Light gray
        self.lidar_ring_color = (0, 100, 0)  # Dark green
        
    def generate_vectorized_map(self,
                                rgb_front: np.ndarray,
                                lidar_data: np.ndarray,
                                traffic_meta: np.ndarray,
                                ego_speed: float,
                                waypoints: np.ndarray = None,
                                map_size: int = 800) -> np.ndarray:
        """
        Generate clean vectorized HD map
        
        Returns clean lines and polygons, not point clouds
        """
        
        # Create canvas
        hd_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        hd_map[:] = self.bg_color
        
        center = map_size // 2
        pixels_per_meter = 10
        
        # 1. Extract and draw road boundary polygon from LiDAR
        road_polygon = self._extract_road_boundary(lidar_data, pixels_per_meter, center, map_size)
        hd_map = self._draw_road_polygon(hd_map, road_polygon)
        
        # 2. Draw LiDAR range rings (like radar display)
        hd_map = self._draw_lidar_rings(hd_map, center, pixels_per_meter)
        
        # 3. Extract vectorized lanes from camera
        lane_vectors = self._extract_lane_vectors(rgb_front)
        hd_map = self._draw_vectorized_lanes(hd_map, lane_vectors, pixels_per_meter, center, map_size)
        
        # 4. Draw road boundaries from LiDAR clustering
        boundaries = self._extract_road_boundaries_from_lidar(lidar_data, pixels_per_meter, center, map_size)
        hd_map = self._draw_boundaries(hd_map, boundaries)
        
        # 5. Draw detected vehicles as clean boxes (not point clusters)
        hd_map = self._draw_clean_vehicles(hd_map, traffic_meta, pixels_per_meter, center, map_size)
        
        # 6. Draw waypoints as smooth path
        if waypoints is not None:
            hd_map = self._draw_smooth_path(hd_map, waypoints, pixels_per_meter, center, map_size)
        
        # 7. Draw ego vehicle with detail
        hd_map = self._draw_detailed_ego_vehicle(hd_map, center, pixels_per_meter)
        
        # 8. Add minimal UI
        self._add_minimal_ui(hd_map, ego_speed)
        
        return hd_map
    
    def _extract_road_boundary(self, lidar_data: np.ndarray,
                               pixels_per_meter: float, center: int,
                               map_size: int) -> Optional[np.ndarray]:
        """Extract road boundary as polygon from ground plane"""
        
        if lidar_data is None or len(lidar_data) == 0:
            return None
        
        points = lidar_data[:, :3]
        
        # Ground plane points
        ground_mask = np.abs(points[:, 2]) < 0.3
        ground_points = points[ground_mask]
        
        if len(ground_points) < 10:
            return None
        
        # Convert to BEV
        bev_points = []
        for p in ground_points:
            px = int(center + p[1] * pixels_per_meter)
            py = int(center - p[0] * pixels_per_meter)
            if 0 <= px < map_size and 0 <= py < map_size:
                bev_points.append([px, py])
        
        if len(bev_points) < 10:
            return None
        
        # Compute convex hull for road boundary
        try:
            bev_array = np.array(bev_points)
            hull = ConvexHull(bev_array)
            polygon = bev_array[hull.vertices]
            return polygon
        except:
            return None
    
    def _draw_road_polygon(self, img: np.ndarray, polygon: Optional[np.ndarray]) -> np.ndarray:
        """Draw road as filled polygon"""
        if polygon is not None:
            cv2.fillPoly(img, [polygon.astype(np.int32)], self.road_color)
        return img
    
    def _draw_lidar_rings(self, img: np.ndarray, center: int, pixels_per_meter: float) -> np.ndarray:
        """Draw concentric range rings like radar"""
        # Draw rings every 10 meters
        for radius_m in [10, 20, 30, 40]:
            radius_px = int(radius_m * pixels_per_meter)
            cv2.circle(img, (center, center), radius_px, self.lidar_ring_color, 1, cv2.LINE_AA)
        return img
    
    def _extract_lane_vectors(self, rgb_image: np.ndarray) -> Dict:
        """
        Extract clean lane vectors from camera
        Returns lines, not messy contours
        """
        
        if rgb_image is None or rgb_image.size == 0:
            return {'left_boundary': [], 'right_boundary': [], 'center_line': [], 'lane_marks': []}
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Edge detection for lane boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply ROI
        roi_mask = np.zeros_like(edges)
        roi_vertices = np.array([[(0, h), (0, h*2//3), (w, h*2//3), (w, h)]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        edges = cv2.bitwise_and(edges, roi_mask)
        
        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=20)
        
        if lines is None:
            return {'left_boundary': [], 'right_boundary': [], 'center_line': [], 'lane_marks': []}
        
        # Separate into left and right lanes
        left_lines = []
        right_lines = []
        center_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter by slope (lanes should be somewhat vertical)
            if abs(slope) < 0.3:  # Too horizontal
                continue
            
            # Classify by position and slope
            midpoint_x = (x1 + x2) / 2
            if midpoint_x < w * 0.4 and slope < 0:  # Left lane
                left_lines.append(line[0])
            elif midpoint_x > w * 0.6 and slope > 0:  # Right lane
                right_lines.append(line[0])
            elif w * 0.4 <= midpoint_x <= w * 0.6:  # Center
                center_lines.append(line[0])
        
        return {
            'left_boundary': left_lines,
            'right_boundary': right_lines,
            'center_line': center_lines,
            'lane_marks': [],  # Additional markings
            'image_height': h,
            'image_width': w
        }
    
    def _draw_vectorized_lanes(self, img: np.ndarray, lane_vectors: Dict,
                               pixels_per_meter: float, center: int,
                               map_size: int) -> np.ndarray:
        """Draw clean vectorized lane lines"""
        
        img_h = lane_vectors.get('image_height', 600)
        img_w = lane_vectors.get('image_width', 800)
        
        def project_line_to_bev(x1, y1, x2, y2):
            """Project camera line to BEV coordinates"""
            points = []
            for x, y in [(x1, y1), (x2, y2)]:
                depth = (img_h - y) / img_h
                if depth < 0.2:
                    continue
                lateral = (x - img_w / 2) / (img_w / 2)
                bev_x = lateral * 15 * depth
                bev_y = (1 - depth) * 30
                px = int(center + bev_x * pixels_per_meter)
                py = int(center - bev_y * pixels_per_meter)
                if 0 <= px < map_size and 0 <= py < map_size:
                    points.append((px, py))
            return points
        
        # Draw left boundary (red/orange)
        for line in lane_vectors.get('left_boundary', []):
            points = project_line_to_bev(*line)
            if len(points) == 2:
                cv2.line(img, points[0], points[1], self.lane_boundary_color, 3, cv2.LINE_AA)
        
        # Draw right boundary (red/orange)
        for line in lane_vectors.get('right_boundary', []):
            points = project_line_to_bev(*line)
            if len(points) == 2:
                cv2.line(img, points[0], points[1], self.lane_boundary_color, 3, cv2.LINE_AA)
        
        # Draw center line (white dashed)
        for line in lane_vectors.get('center_line', []):
            points = project_line_to_bev(*line)
            if len(points) == 2:
                self._draw_dashed_line(img, points[0], points[1], self.lane_marking_color, 2)
        
        return img
    
    def _draw_dashed_line(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                          color: Tuple[int, int, int], thickness: int):
        """Draw dashed line"""
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        if dist < 1:
            return
        
        num_dashes = int(dist / 20)  # Dash every 20 pixels
        for i in range(num_dashes):
            if i % 2 == 0:  # Draw every other segment
                t1 = i / num_dashes
                t2 = min((i + 0.5) / num_dashes, 1.0)
                
                p1 = (int(pt1[0] + t1 * (pt2[0] - pt1[0])),
                      int(pt1[1] + t1 * (pt2[1] - pt1[1])))
                p2 = (int(pt1[0] + t2 * (pt2[0] - pt1[0])),
                      int(pt1[1] + t2 * (pt2[1] - pt1[1])))
                
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    
    def _extract_road_boundaries_from_lidar(self, lidar_data: np.ndarray,
                                           pixels_per_meter: float, center: int,
                                           map_size: int) -> List:
        """Extract road edge boundaries from LiDAR using clustering"""
        
        if lidar_data is None or len(lidar_data) == 0:
            return []
        
        points = lidar_data[:, :3]
        
        # Get non-ground points (potential obstacles/boundaries)
        boundary_mask = (points[:, 2] > 0.3) & (points[:, 2] < 2.0)
        boundary_points = points[boundary_mask]
        
        if len(boundary_points) < 10:
            return []
        
        # Convert to BEV
        bev_points = []
        for p in boundary_points:
            px = center + p[1] * pixels_per_meter
            py = center - p[0] * pixels_per_meter
            if 0 <= px < map_size and 0 <= py < map_size:
                bev_points.append([px, py])
        
        if len(bev_points) < 10:
            return []
        
        # Cluster to find road edges
        try:
            clustering = DBSCAN(eps=20, min_samples=5).fit(bev_points)
            labels = clustering.labels_
            
            boundaries = []
            for label in set(labels):
                if label == -1:  # Noise
                    continue
                cluster_points = np.array([bev_points[i] for i in range(len(bev_points)) if labels[i] == label])
                if len(cluster_points) > 5:
                    boundaries.append(cluster_points)
            
            return boundaries
        except:
            return []
    
    def _draw_boundaries(self, img: np.ndarray, boundaries: List) -> np.ndarray:
        """Draw road boundaries as lines"""
        for boundary in boundaries:
            if len(boundary) > 2:
                # Fit line to boundary points
                pts = boundary.astype(np.int32)
                cv2.polylines(img, [pts], False, self.lane_boundary_color, 2, cv2.LINE_AA)
        return img
    
    def _draw_clean_vehicles(self, img: np.ndarray, traffic_meta: np.ndarray,
                            pixels_per_meter: float, center: int,
                            map_size: int) -> np.ndarray:
        """Draw vehicles as clean oriented rectangles"""
        
        if traffic_meta is None:
            return img
        
        if len(traffic_meta.shape) == 2:
            traffic_meta = traffic_meta.reshape(20, 20, 7)
        
        for i in range(20):
            for j in range(20):
                cell = traffic_meta[i, j]
                
                # Only draw high-confidence detections
                if cell[4] > 0.6:  # Vehicle
                    grid_x = (j - 10) * 2.0
                    grid_y = (10 - i) * 2.0
                    obj_x = grid_x + cell[0]
                    obj_y = grid_y + cell[1]
                    
                    px = int(center + obj_x * pixels_per_meter)
                    py = int(center - obj_y * pixels_per_meter)
                    
                    if 0 <= px < map_size and 0 <= py < map_size:
                        self._draw_vehicle_box(img, px, py, cell[3], (0, 0, 200), (18, 9))
                
                elif cell[5] > 0.6:  # Pedestrian
                    grid_x = (j - 10) * 2.0
                    grid_y = (10 - i) * 2.0
                    obj_x = grid_x + cell[0]
                    obj_y = grid_y + cell[1]
                    
                    px = int(center + obj_x * pixels_per_meter)
                    py = int(center - obj_y * pixels_per_meter)
                    
                    if 0 <= px < map_size and 0 <= py < map_size:
                        cv2.circle(img, (px, py), 5, (0, 150, 0), -1)
                        cv2.circle(img, (px, py), 6, (255, 255, 255), 1)
        
        return img
    
    def _draw_vehicle_box(self, img: np.ndarray, x: int, y: int,
                         yaw: float, color: Tuple[int, int, int],
                         size: Tuple[int, int]):
        """Draw clean oriented vehicle box"""
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
        cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
    
    def _draw_smooth_path(self, img: np.ndarray, waypoints: np.ndarray,
                         pixels_per_meter: float, center: int,
                         map_size: int) -> np.ndarray:
        """Draw waypoint path as smooth curve"""
        if len(waypoints.shape) == 1:
            waypoints = waypoints.reshape(-1, 2)
        
        points = []
        for wp in waypoints:
            px = int(center + wp[0] * pixels_per_meter)
            py = int(center - wp[1] * pixels_per_meter)
            if 0 <= px < map_size and 0 <= py < map_size:
                points.append([px, py])
        
        if len(points) > 2:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
        
        return img
    
    def _draw_detailed_ego_vehicle(self, img: np.ndarray, center: int,
                                   pixels_per_meter: float) -> np.ndarray:
        """Draw detailed ego vehicle like Tesla visualization"""
        
        # Vehicle body (realistic proportions)
        length = int(4.5 * pixels_per_meter)  # 4.5m car
        width = int(1.8 * pixels_per_meter)   # 1.8m wide
        
        corners = np.array([
            [center - width//2, center - length//2],
            [center + width//2, center - length//2],
            [center + width//2, center + length//2],
            [center - width//2, center + length//2]
        ], dtype=np.int32)
        
        # Draw vehicle
        cv2.fillPoly(img, [corners], self.ego_color)
        cv2.polylines(img, [corners], True, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw windshield (front indicator)
        windshield_y = center - length//2 + 5
        cv2.line(img, (center - width//3, windshield_y),
                (center + width//3, windshield_y), (100, 100, 255), 2, cv2.LINE_AA)
        
        # Draw heading arrow
        arrow_start = center - length//2
        arrow_end = arrow_start - 15
        cv2.arrowedLine(img, (center, arrow_start), (center, arrow_end),
                       (255, 255, 255), 2, tipLength=0.3)
        
        return img
    
    def _add_minimal_ui(self, img: np.ndarray, ego_speed: float):
        """Add minimal UI overlay"""
        cv2.putText(img, f'{ego_speed*3.6:.1f} km/h', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
