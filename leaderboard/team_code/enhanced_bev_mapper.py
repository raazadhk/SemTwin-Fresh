"""
Enhanced BEV Semantic Mapper with Rich Object Detection
Improved semantic segmentation with 15+ classes and better tracking
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import defaultdict


class EnhancedBEVMapper:
    """
    Enhanced BEV semantic mapper with rich object detection
    
    Features:
    - 15+ semantic classes
    - Object instance tracking
    - Temporal consistency
    - Point cloud clustering for objects
    - Traffic infrastructure detection
    """
    
    # Extended semantic classes
    CLASSES = {
        0: 'road',
        1: 'lane_marking',
        2: 'sidewalk',
        3: 'crosswalk',
        4: 'vehicle',
        5: 'truck',
        6: 'bus',
        7: 'motorcycle',
        8: 'bicycle',
        9: 'pedestrian',
        10: 'traffic_light',
        11: 'traffic_sign',
        12: 'pole',
        13: 'barrier',
        14: 'vegetation',
        15: 'building',
        16: 'fence',
        17: 'other'
    }
    
    # Enhanced color scheme (BGR)
    COLORS = {
        0: (128, 0, 128),      # road - purple
        1: (0, 165, 255),      # lane - orange
        2: (200, 200, 200),    # sidewalk - light gray
        3: (0, 255, 255),      # crosswalk - cyan
        4: (0, 0, 255),        # vehicle - red
        5: (0, 0, 180),        # truck - dark red
        6: (0, 100, 200),      # bus - orange-red
        7: (128, 0, 255),      # motorcycle - purple-red
        8: (255, 0, 255),      # bicycle - magenta
        9: (255, 100, 0),      # pedestrian - blue-orange
        10: (0, 255, 0),       # traffic light - green
        11: (0, 255, 150),     # traffic sign - light green
        12: (150, 150, 150),   # pole - gray
        13: (100, 200, 255),   # barrier - yellow
        14: (34, 139, 34),     # vegetation - green
        15: (80, 80, 80),      # building - dark gray
        16: (120, 120, 120),   # fence - medium gray
        17: (50, 50, 50),      # other - very dark
    }
    
    def __init__(self, 
                 bev_size=(800, 800),
                 bev_range=50.0,
                 pixels_per_meter=16,
                 point_size=2,
                 enable_tracking=True):
        """
        Args:
            bev_size: Output image size
            bev_range: Range in meters
            pixels_per_meter: Resolution
            point_size: Rendering point size
            enable_tracking: Enable temporal object tracking
        """
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.pixels_per_meter = pixels_per_meter
        self.point_size = point_size
        self.enable_tracking = enable_tracking
        
        # Tracking state
        self.tracked_objects = {}
        self.next_object_id = 0
        self.prev_semantic_map = None
        
        # Background
        self.background_color = (40, 40, 40)
        
    def generate_bev_semantic_map(self,
                                  lidar_data: np.ndarray,
                                  rgb_front: Optional[np.ndarray] = None,
                                  traffic_meta: Optional[np.ndarray] = None,
                                  bev_feature: Optional[np.ndarray] = None) -> Dict:
        """
        Generate enhanced BEV semantic map with rich object detection
        """
        H, W = self.bev_size
        
        # Initialize maps
        semantic_map = np.ones((H, W), dtype=np.uint8) * 17  # default: other
        instance_map = np.zeros((H, W), dtype=np.int32)  # object IDs
        confidence_map = np.zeros((H, W), dtype=np.float32)
        
        # 1. Cluster LiDAR points into objects
        if lidar_data is not None and len(lidar_data) > 0:
            object_clusters = self._cluster_lidar_points(lidar_data)
            
            # 2. Classify each cluster
            classified_objects = self._classify_clusters(object_clusters, traffic_meta)
            
            # 3. Track objects over time
            if self.enable_tracking:
                tracked_objects = self._track_objects(classified_objects)
            else:
                tracked_objects = classified_objects
            
            # 4. Render to semantic map
            semantic_map, instance_map, confidence_map = self._render_objects(
                tracked_objects, H, W, semantic_map, instance_map, confidence_map
            )
        
        # 5. Add detected objects from traffic_meta (refined)
        if traffic_meta is not None:
            semantic_map, instance_map = self._add_traffic_meta_objects(
                traffic_meta, H, W, semantic_map, instance_map
            )
        
        # 6. Render to RGB
        rendered = self._render_semantic_map(semantic_map, instance_map)
        
        # Store for temporal consistency
        self.prev_semantic_map = semantic_map.copy()
        
        return {
            'semantic_map': semantic_map,
            'instance_map': instance_map,
            'confidence': confidence_map,
            'rendered': rendered,
            'tracked_objects': tracked_objects if self.enable_tracking else [],
            'class_names': self.CLASSES
        }
    
    def _cluster_lidar_points(self, lidar_data: np.ndarray, 
                              eps=0.5, min_points=10) -> List[Dict]:
        """
        Cluster LiDAR points into object instances using DBSCAN-like clustering
        """
        clusters = []
        
        # Separate ground and non-ground points
        ground_mask = (lidar_data[:, 2] > -2.3) & (lidar_data[:, 2] < -1.3)
        non_ground = lidar_data[~ground_mask]
        ground = lidar_data[ground_mask]
        
        # Process ground points (road, sidewalk, lanes)
        clusters.append({
            'points': ground,
            'centroid': np.mean(ground[:, :3], axis=0) if len(ground) > 0 else np.zeros(3),
            'type': 'ground',
            'size': ground.shape[0]
        })
        
        # Cluster non-ground points (objects)
        if len(non_ground) > 0:
            # Simple spatial clustering
            object_clusters = self._simple_clustering(non_ground[:, :3], eps, min_points)
            
            for cluster_points in object_clusters:
                if len(cluster_points) >= min_points:
                    centroid = np.mean(cluster_points, axis=0)
                    bbox = self._compute_bbox(cluster_points)
                    
                    clusters.append({
                        'points': cluster_points,
                        'centroid': centroid,
                        'bbox': bbox,
                        'type': 'object',
                        'size': len(cluster_points)
                    })
        
        return clusters
    
    def _simple_clustering(self, points: np.ndarray, eps: float, min_points: int) -> List:
        """Simple DBSCAN-like clustering"""
        clusters = []
        visited = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if visited[i]:
                continue
            
            # Find neighbors
            distances = np.linalg.norm(points - points[i], axis=1)
            neighbors = np.where(distances < eps)[0]
            
            if len(neighbors) >= min_points:
                # Start new cluster
                cluster = []
                queue = list(neighbors)
                
                while queue:
                    idx = queue.pop(0)
                    if visited[idx]:
                        continue
                    
                    visited[idx] = True
                    cluster.append(points[idx])
                    
                    # Expand cluster
                    distances = np.linalg.norm(points - points[idx], axis=1)
                    new_neighbors = np.where(distances < eps)[0]
                    
                    for n in new_neighbors:
                        if not visited[n]:
                            queue.append(n)
                
                if len(cluster) > 0:
                    clusters.append(np.array(cluster))
        
        return clusters
    
    def _compute_bbox(self, points: np.ndarray) -> Dict:
        """Compute 3D bounding box"""
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        
        center = (min_pt + max_pt) / 2
        size = max_pt - min_pt
        
        return {
            'center': center,
            'size': size,
            'min': min_pt,
            'max': max_pt
        }
    
    def _classify_clusters(self, clusters: List[Dict], 
                          traffic_meta: Optional[np.ndarray]) -> List[Dict]:
        """
        Classify each cluster into semantic classes
        """
        classified = []
        
        for cluster in clusters:
            if cluster['type'] == 'ground':
                # Ground points - classify by location and intensity
                cluster['class_id'] = 0  # road (will be refined later)
                classified.append(cluster)
                continue
            
            # Object classification based on size and shape
            bbox = cluster['bbox']
            size = bbox['size']
            height = size[2]
            width = max(size[0], size[1])
            
            # Classification heuristics
            if height > 3.0:  # Tall objects
                if width > 2.0:
                    cluster['class_id'] = 15  # building
                else:
                    cluster['class_id'] = 12  # pole
                    
            elif height > 2.0:  # Medium height
                if width > 5.0:
                    cluster['class_id'] = 6  # bus
                elif width > 3.0:
                    cluster['class_id'] = 5  # truck
                elif width > 1.5:
                    cluster['class_id'] = 4  # vehicle
                else:
                    cluster['class_id'] = 10  # traffic light/sign
                    
            elif height > 1.0:  # Human height
                if width < 1.0:
                    cluster['class_id'] = 9  # pedestrian
                elif width < 2.0:
                    cluster['class_id'] = 8  # bicycle
                else:
                    cluster['class_id'] = 4  # vehicle
                    
            elif height > 0.3:  # Low objects
                if width > 1.0:
                    cluster['class_id'] = 13  # barrier
                else:
                    cluster['class_id'] = 7  # motorcycle
            else:
                cluster['class_id'] = 17  # other
            
            classified.append(cluster)
        
        return classified
    
    def _track_objects(self, current_objects: List[Dict]) -> List[Dict]:
        """
        Track objects across frames for temporal consistency
        """
        tracked = []
        matched_ids = set()
        
        # Match current objects with tracked objects
        for obj in current_objects:
            if obj['type'] == 'ground':
                tracked.append(obj)
                continue
            
            centroid = obj['centroid']
            best_match = None
            best_dist = float('inf')
            
            # Find closest tracked object
            for obj_id, tracked_obj in self.tracked_objects.items():
                if obj_id in matched_ids:
                    continue
                
                dist = np.linalg.norm(centroid - tracked_obj['centroid'])
                
                # If close enough and same class
                if dist < 2.0 and dist < best_dist:
                    if obj['class_id'] == tracked_obj['class_id']:
                        best_match = obj_id
                        best_dist = dist
            
            if best_match is not None:
                # Update existing object
                obj['id'] = best_match
                obj['track_age'] = self.tracked_objects[best_match]['track_age'] + 1
                matched_ids.add(best_match)
                self.tracked_objects[best_match] = obj
            else:
                # New object
                obj['id'] = self.next_object_id
                obj['track_age'] = 0
                self.tracked_objects[obj['id']] = obj
                self.next_object_id += 1
            
            tracked.append(obj)
        
        # Remove old objects (not seen for 5 frames)
        to_remove = []
        for obj_id in self.tracked_objects:
            if obj_id not in matched_ids:
                self.tracked_objects[obj_id]['track_age'] -= 1
                if self.tracked_objects[obj_id]['track_age'] < -5:
                    to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
        
        return tracked
    
    def _render_objects(self, objects: List[Dict], H: int, W: int,
                       semantic_map: np.ndarray, instance_map: np.ndarray,
                       confidence_map: np.ndarray) -> Tuple:
        """
        Render classified objects to semantic and instance maps
        """
        for obj in objects:
            class_id = obj['class_id']
            points = obj['points']
            obj_id = obj.get('id', 0)
            
            # Render each point
            for point in points:
                x, y, z = point[:3]
                
                # Convert to pixel coordinates
                pixel_x = int(W/2 + y * self.pixels_per_meter)
                pixel_y = int(H - (x + self.bev_range/2) * self.pixels_per_meter)
                
                if 0 <= pixel_x < W and 0 <= pixel_y < H:
                    semantic_map[pixel_y, pixel_x] = class_id
                    instance_map[pixel_y, pixel_x] = obj_id
                    confidence_map[pixel_y, pixel_x] = min(1.0, obj.get('track_age', 0) / 10.0)
        
        return semantic_map, instance_map, confidence_map
    
    def _add_traffic_meta_objects(self, traffic_meta: np.ndarray, H: int, W: int,
                                  semantic_map: np.ndarray, 
                                  instance_map: np.ndarray) -> Tuple:
        """
        Add high-confidence objects from traffic_meta
        """
        if traffic_meta.shape != (20, 20, 7):
            traffic_meta = traffic_meta.reshape(20, 20, 7)
        
        for i in range(20):
            for j in range(20):
                meta = traffic_meta[i, j]
                x, y = meta[0], meta[1]
                bbox_x, bbox_y = meta[3], meta[4]
                label = int(meta[6])
                
                if label == 0 and abs(x) < 0.01:
                    continue
                
                # Map label to enhanced classes
                if label == 1:  # vehicle
                    # Distinguish by size
                    if bbox_x > 6.0:
                        class_id = 6  # bus
                    elif bbox_x > 4.0:
                        class_id = 5  # truck
                    else:
                        class_id = 4  # vehicle
                elif label == 2:
                    class_id = 9  # pedestrian
                elif label == 3:
                    class_id = 8  # bicycle/motorcycle
                else:
                    continue
                
                # Render bounding box
                pixel_x = int(W/2 + y * self.pixels_per_meter)
                pixel_y = int(H - (x + self.bev_range/2) * self.pixels_per_meter)
                
                bbox_w = max(4, int(bbox_y * self.pixels_per_meter))
                bbox_h = max(4, int(bbox_x * self.pixels_per_meter))
                
                y1 = max(0, pixel_y - bbox_h//2)
                y2 = min(H, pixel_y + bbox_h//2)
                x1 = max(0, pixel_x - bbox_w//2)
                x2 = min(W, pixel_x + bbox_w//2)
                
                semantic_map[y1:y2, x1:x2] = class_id
                instance_map[y1:y2, x1:x2] = 1000 + i * 20 + j  # unique ID
        
        return semantic_map, instance_map
    
    def _render_semantic_map(self, semantic_map: np.ndarray, 
                            instance_map: np.ndarray) -> np.ndarray:
        """
        Render semantic map to RGB with instance boundaries
        """
        H, W = semantic_map.shape
        rendered = np.ones((H, W, 3), dtype=np.uint8) * np.array(self.background_color)
        
        # Render semantic colors
        for class_id, color in self.COLORS.items():
            mask = semantic_map == class_id
            rendered[mask] = color
        
        # Add instance boundaries (white outlines)
        boundaries = self._find_instance_boundaries(instance_map)
        rendered[boundaries] = (255, 255, 255)
        
        return rendered
    
    def _find_instance_boundaries(self, instance_map: np.ndarray) -> np.ndarray:
        """Find boundaries between instances"""
        # Sobel edge detection on instance map
        dx = cv2.Sobel(instance_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(instance_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(dx**2 + dy**2)
        
        boundaries = edges > 0.5
        return boundaries
    
    def create_visualization_with_overlay(self, result: Dict, 
                                         velocity: float = 0.0) -> np.ndarray:
        """
        Create enhanced visualization with object labels and tracking info
        """
        # Get base rendered image
        base_rendered = result['rendered']
        H, W = base_rendered.shape[:2]
        
        # Create new image for overlay
        rendered = np.zeros((H, W, 3), dtype=np.uint8)
        rendered[:] = base_rendered  # Copy pixels
        
        semantic_map = result['semantic_map']
        tracked_objects = result.get('tracked_objects', [])
        
        # Add grid
        for i in range(0, H, 80):
            cv2.line(rendered, (0, i), (W, i), (60, 60, 60), 1)
        for j in range(0, W, 80):
            cv2.line(rendered, (j, 0), (j, H), (60, 60, 60), 1)
        cv2.line(rendered, (W//2, 0), (W//2, H), (80, 80, 80), 1)
        
        # Add ego vehicle
        ego_x, ego_y = W//2, int(H * 0.95)
        tri = np.array([[ego_x, ego_y-20], [ego_x-10, ego_y], [ego_x+10, ego_y]])
        cv2.fillPoly(rendered, [tri], (0, 255, 255))
        cv2.polylines(rendered, [tri], True, (255, 255, 255), 2)
        cv2.arrowedLine(rendered, (ego_x, ego_y-20), (ego_x, ego_y-35),
                       (255, 255, 255), 2, tipLength=0.3)
        
        # Add object labels and IDs
        for obj in tracked_objects:
            if obj['type'] == 'ground':
                continue
            
            centroid = obj['centroid']
            pixel_x = int(W/2 + centroid[1] * self.pixels_per_meter)
            pixel_y = int(H - (centroid[0] + self.bev_range/2) * self.pixels_per_meter)
            
            if 0 <= pixel_x < W and 0 <= pixel_y < H:
                # Draw object ID
                obj_id = obj.get('id', 0)
                class_name = self.CLASSES[obj['class_id']][:3].upper()
                label = f"{class_name}#{obj_id}"
                
                cv2.putText(rendered, label, (pixel_x-15, pixel_y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Draw center point
                cv2.circle(rendered, (pixel_x, pixel_y), 3, (255, 255, 255), -1)
        
        # Statistics panel
        panel = self._create_enhanced_stats(result, velocity)
        rendered[:panel.shape[0], :panel.shape[1]] = cv2.addWeighted(
            rendered[:panel.shape[0], :panel.shape[1]], 0.3,
            panel, 0.7, 0
        )
        
        return rendered
    
    def _add_grid(self, img, spacing=80, color=(60, 60, 60)):
        img = img.copy()  # Make writable copy
        H, W = img.shape[:2]
        for i in range(0, H, spacing):
            cv2.line(img, (0, i), (W, i), color, 1)
        for j in range(0, W, spacing):
            cv2.line(img, (j, 0), (j, H), color, 1)
        cv2.line(img, (W//2, 0), (W//2, H), (80, 80, 80), 1)
        return img
    
    def _add_ego_vehicle(self, img):
        img = img.copy()
        H, W = img.shape[:2]
        ego_x, ego_y = W//2, int(H * 0.95)
        tri = np.array([[ego_x, ego_y-20], [ego_x-10, ego_y], [ego_x+10, ego_y]])
        cv2.fillPoly(img, [tri], (0, 255, 255))
        cv2.polylines(img, [tri], True, (255, 255, 255), 2)
        cv2.arrowedLine(img, (ego_x, ego_y-20), (ego_x, ego_y-35),
                       (255, 255, 255), 2, tipLength=0.3)
        return img
    
    def _create_enhanced_stats(self, result, velocity):
        """Enhanced statistics with object counts by type"""
        panel = np.ones((200, 280, 3), dtype=np.uint8) * 20
        semantic_map = result['semantic_map']
        
        y = 20
        cv2.putText(panel, 'Enhanced Semantic Map', (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y += 25
        
        cv2.putText(panel, f'Speed: {velocity:.1f} m/s', (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y += 25
        
        # Count objects by type
        counts = defaultdict(int)
        for class_id in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            counts[class_id] = int(np.sum(semantic_map == class_id))
        
        for class_id, count in counts.items():
            if count > 50:  # Only show significant detections
                name = self.CLASSES[class_id][:10]
                color = self.COLORS[class_id]
                cv2.rectangle(panel, (10, y-10), (25, y), color, -1)
                cv2.putText(panel, f"{name}: {count}", (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                y += 16
        
        return panel


# Test
if __name__ == "__main__":
    print("Testing Enhanced BEV Mapper...")
    
    # Create realistic test data
    np.random.seed(42)
    
    # Generate LiDAR with multiple objects
    lidar_data = []
    
    # Ground
    for _ in range(3000):
        x = np.random.uniform(0, 40)
        y = np.random.uniform(-8, 8)
        z = np.random.uniform(-2.0, -1.7)
        intensity = np.random.uniform(0.3, 0.7)
        lidar_data.append([x, y, z, intensity])
    
    # Vehicle 1
    for _ in range(200):
        x = 15 + np.random.uniform(-2, 2)
        y = -2 + np.random.uniform(-0.9, 0.9)
        z = np.random.uniform(-0.5, 1.5)
        intensity = 0.5
        lidar_data.append([x, y, z, intensity])
    
    # Pedestrians
    for _ in range(80):
        x = 10 + np.random.uniform(-0.3, 0.3)
        y = 3 + np.random.uniform(-0.3, 0.3)
        z = np.random.uniform(0, 1.7)
        intensity = 0.4
        lidar_data.append([x, y, z, intensity])
    
    # Pole
    for _ in range(50):
        x = 20
        y = 5
        z = np.random.uniform(0, 4)
        intensity = 0.6
        lidar_data.append([x, y, z, intensity])
    
    lidar_data = np.array(lidar_data)
    
    # Traffic meta
    traffic_meta = np.zeros((20, 20, 7))
    traffic_meta[12, 9] = [15.0, -2.0, 0.0, 4.0, 1.8, 10.0, 1]
    traffic_meta[11, 12] = [10.0, 3.0, 0.0, 0.5, 0.5, 2.0, 2]
    
    # Generate
    mapper = EnhancedBEVMapper(
        bev_size=(800, 800),
        bev_range=50.0,
        pixels_per_meter=16,
        enable_tracking=True
    )
    
    result = mapper.generate_bev_semantic_map(
        lidar_data=lidar_data,
        traffic_meta=traffic_meta
    )
    
    viz = mapper.create_visualization_with_overlay(result, velocity=12.5)
    
    print(f"✓ Semantic map: {result['semantic_map'].shape}")
    print(f"✓ Classes found: {np.unique(result['semantic_map'])}")
    print(f"✓ Tracked objects: {len(result['tracked_objects'])}")
    
    cv2.imwrite('/home/claude/enhanced_bev_test.png', viz)
    print("✓ Saved: /home/claude/enhanced_bev_test.png")
    
    print("\n✓ Enhanced BEV Mapper ready!")
