#!/usr/bin/env python3
"""
HD Map Builder - Cloud-Based HD Map Construction from CARLA Sensor Data

This system receives sensor data from the InterFuser agent and builds HD maps
in real-time, similar to how mapping companies create maps.

Author: HD Map Builder System
Date: December 2025
"""

import numpy as np
import cv2
import json
import os
import time
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
import pickle


class PointCloudMapBuilder:
    """
    Accumulates LiDAR point clouds and builds a global 3D map
    """
    
    def __init__(self, voxel_size=0.1, max_points=10_000_000):
        """
        Initialize point cloud map builder
        
        Args:
            voxel_size: Voxel grid size for downsampling (meters)
            max_points: Maximum number of points to keep in memory
        """
        self.voxel_size = voxel_size
        self.max_points = max_points
        
        # Global point cloud storage (voxel grid)
        self.voxel_map = {}  # {(vx, vy, vz): point_data}
        
        # Statistics
        self.total_scans_processed = 0
        self.total_points_added = 0
        
        print(f"✓ Point Cloud Map Builder initialized (voxel_size={voxel_size}m)")
    
    def add_lidar_scan(self, points, position, orientation):
        """
        Add a LiDAR scan to the global map
        
        Args:
            points: Nx3 array of points in vehicle frame
            position: [x, y, z] vehicle position in world frame
            orientation: [roll, pitch, yaw] vehicle orientation (radians)
        """
        if points.shape[0] == 0:
            return
        
        # Transform points to global frame
        global_points = self._transform_to_global(points, position, orientation)
        
        # Filter points (remove ground, outliers, etc.)
        filtered_points = self._filter_points(global_points)
        
        # Add to voxel map
        points_added = 0
        for point in filtered_points:
            voxel_key = self._get_voxel_key(point)
            
            if voxel_key not in self.voxel_map:
                # New voxel
                self.voxel_map[voxel_key] = {
                    'centroid': point.copy(),
                    'count': 1,
                    'intensity': 0
                }
                points_added += 1
            else:
                # Update existing voxel (moving average)
                voxel = self.voxel_map[voxel_key]
                count = voxel['count']
                voxel['centroid'] = (voxel['centroid'] * count + point) / (count + 1)
                voxel['count'] += 1
        
        self.total_scans_processed += 1
        self.total_points_added += points_added
        
        # Downsample if too many points
        if len(self.voxel_map) > self.max_points:
            self._downsample_map()
    
    def _transform_to_global(self, points, position, orientation):
        """Transform points from vehicle frame to global frame"""
        roll, pitch, yaw = orientation
        
        # Rotation matrix (yaw only for CARLA, as roll/pitch are usually 0)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation and translation
        global_points = points @ R.T + np.array(position)
        
        return global_points
    
    def _filter_points(self, points):
        """Filter out ground points and outliers"""
        # Simple ground removal: points below -1.5m relative to sensor
        # (CARLA sensor is at z=2.5, so ground is around z=0-1m)
        ground_threshold = 0.5
        filtered = points[points[:, 2] > ground_threshold]
        
        # Remove points too far away (likely noise)
        distances = np.linalg.norm(filtered[:, :2], axis=1)
        filtered = filtered[distances < 100]  # 100m max range
        
        return filtered
    
    def _get_voxel_key(self, point):
        """Get voxel grid key for a point"""
        vx = int(np.floor(point[0] / self.voxel_size))
        vy = int(np.floor(point[1] / self.voxel_size))
        vz = int(np.floor(point[2] / self.voxel_size))
        return (vx, vy, vz)
    
    def _downsample_map(self):
        """Downsample map by removing least observed voxels"""
        if len(self.voxel_map) <= self.max_points:
            return
        
        # Sort by observation count and keep top max_points
        sorted_voxels = sorted(
            self.voxel_map.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )
        
        self.voxel_map = dict(sorted_voxels[:self.max_points])
        print(f"  Downsampled to {len(self.voxel_map)} voxels")
    
    def get_point_cloud(self):
        """Get current point cloud as Nx3 array"""
        if not self.voxel_map:
            return np.array([]).reshape(0, 3)
        
        points = np.array([voxel['centroid'] for voxel in self.voxel_map.values()])
        return points
    
    def save_point_cloud(self, filepath):
        """Save point cloud to file (.pcd or .ply format)"""
        points = self.get_point_cloud()
        
        if filepath.endswith('.npy'):
            np.save(filepath, points)
            print(f"✓ Saved {len(points)} points to {filepath}")
        
        elif filepath.endswith('.pcd'):
            self._save_pcd(points, filepath)
        
        elif filepath.endswith('.ply'):
            self._save_ply(points, filepath)
        
        else:
            raise ValueError(f"Unsupported format: {filepath}")
    
    def _save_pcd(self, points, filepath):
        """Save as PCD format (Point Cloud Data)"""
        with open(filepath, 'w') as f:
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            f.write(f"WIDTH {len(points)}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {len(points)}\n")
            f.write("DATA ascii\n")
            
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        print(f"✓ Saved {len(points)} points to {filepath} (PCD format)")
    
    def _save_ply(self, points, filepath):
        """Save as PLY format"""
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        print(f"✓ Saved {len(points)} points to {filepath} (PLY format)")
    
    def get_statistics(self):
        """Get statistics about the map"""
        return {
            'total_scans_processed': self.total_scans_processed,
            'total_points_added': self.total_points_added,
            'current_voxels': len(self.voxel_map),
            'voxel_size': self.voxel_size,
            'max_points': self.max_points
        }


class LaneExtractor:
    """
    Extracts lane boundaries from semantic segmentation
    """
    
    def __init__(self):
        """Initialize lane extractor"""
        self.lane_segments = []  # List of lane segments
        self.lane_points_buffer = []  # Buffer for accumulating lane points
        
        print("✓ Lane Extractor initialized")
    
    def process_frame(self, rgb_image, seg_image, position, orientation, camera_params):
        """
        Process a frame to extract lane markings
        
        Args:
            rgb_image: RGB image (HxWx3)
            seg_image: Semantic segmentation image (HxWx3, class in channel 2)
            position: [x, y, z] camera position
            orientation: [roll, pitch, yaw] camera orientation
            camera_params: Dict with 'width', 'height', 'fov'
        """
        # Extract lane marking pixels (class 6: RoadLine)
        seg_class_ids = seg_image[:, :, 2]
        lane_mask = (seg_class_ids == 6)
        
        if not np.any(lane_mask):
            return []
        
        # Get lane pixel coordinates
        lane_pixels = np.argwhere(lane_mask)  # (N, 2) array of (row, col)
        
        # Convert to 3D points (project to ground plane)
        lane_points_3d = self._project_to_3d(
            lane_pixels, position, orientation, camera_params
        )
        
        # Add to buffer
        self.lane_points_buffer.extend(lane_points_3d)
        
        # Fit lanes periodically
        if len(self.lane_points_buffer) > 500:
            lanes = self._fit_lanes(self.lane_points_buffer)
            self.lane_segments.extend(lanes)
            self.lane_points_buffer = []
            return lanes
        
        return []
    
    def _project_to_3d(self, pixels, position, orientation, camera_params):
        """Project 2D pixels to 3D ground points"""
        H = camera_params['height']
        W = camera_params['width']
        fov = camera_params['fov']
        
        # Camera intrinsics
        fx = W / (2 * np.tan(np.radians(fov) / 2))
        fy = fx
        cx = W / 2
        cy = H / 2
        
        points_3d = []
        
        for pixel in pixels:
            row, col = pixel
            
            # Pixel to normalized image coordinates
            x_norm = (col - cx) / fx
            y_norm = (row - cy) / fy
            
            # Assume ground plane at z=0, camera at height position[2]
            # Ray: origin = position, direction = [x_norm, y_norm, 1] (normalized)
            z_camera = position[2]
            
            # Intersection with ground plane (z=0)
            if y_norm < 0:  # Ray pointing downward
                t = z_camera / (-y_norm)
                x_3d = position[0] + t * x_norm * np.cos(orientation[2]) - t * 1 * np.sin(orientation[2])
                y_3d = position[1] + t * x_norm * np.sin(orientation[2]) + t * 1 * np.cos(orientation[2])
                
                points_3d.append([x_3d, y_3d, 0.0])
        
        return points_3d
    
    def _fit_lanes(self, points):
        """Fit lane curves to points using clustering and spline fitting"""
        if len(points) < 10:
            return []
        
        points_array = np.array(points)
        
        # Cluster points into separate lanes using DBSCAN
        clustering = DBSCAN(eps=2.0, min_samples=5).fit(points_array[:, :2])
        labels = clustering.labels_
        
        lanes = []
        
        for label in set(labels):
            if label == -1:  # Noise
                continue
            
            cluster_points = points_array[labels == label]
            
            if len(cluster_points) < 10:
                continue
            
            # Sort points by x coordinate
            sorted_indices = np.argsort(cluster_points[:, 0])
            sorted_points = cluster_points[sorted_indices]
            
            # Fit spline
            try:
                tck, u = splprep([sorted_points[:, 0], sorted_points[:, 1]], s=1.0, k=3)
                u_new = np.linspace(0, 1, 50)
                lane_curve = splev(u_new, tck)
                
                lane = {
                    'points': np.array([lane_curve[0], lane_curve[1], np.zeros_like(lane_curve[0])]).T,
                    'type': 'lane_marking',
                    'num_points': len(cluster_points)
                }
                
                lanes.append(lane)
            except:
                pass
        
        return lanes
    
    def get_all_lanes(self):
        """Get all extracted lane segments"""
        return self.lane_segments
    
    def save_lanes(self, filepath):
        """Save lanes to JSON file"""
        lanes_data = {
            'total_lanes': len(self.lane_segments),
            'lanes': [
                {
                    'type': lane['type'],
                    'points': lane['points'].tolist(),
                    'num_points': lane['num_points']
                }
                for lane in self.lane_segments
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(lanes_data, f, indent=2)
        
        print(f"✓ Saved {len(self.lane_segments)} lane segments to {filepath}")


class TrafficElementMapper:
    """
    Maps traffic signs and lights from semantic segmentation
    """
    
    def __init__(self, merge_distance=5.0):
        """
        Initialize traffic element mapper
        
        Args:
            merge_distance: Distance threshold to merge duplicate detections (meters)
        """
        self.merge_distance = merge_distance
        self.traffic_lights = {}  # {id: {position, observations, state_history}}
        self.traffic_signs = {}   # {id: {position, observations, type}}
        
        self.next_light_id = 0
        self.next_sign_id = 0
        
        print(f"✓ Traffic Element Mapper initialized (merge_distance={merge_distance}m)")
    
    def process_frame(self, seg_image, tracked_objects, position, orientation):
        """
        Process frame to detect and map traffic elements
        
        Args:
            seg_image: Semantic segmentation image
            tracked_objects: Dict of tracked objects from tracking system
            position: [x, y, z] vehicle position
            orientation: [roll, pitch, yaw] vehicle orientation
        """
        # Extract traffic elements from tracked objects
        for track_id, obj in tracked_objects.items():
            class_name = obj.get('class_name', '')
            
            if class_name == 'traffic_light':
                self._add_traffic_light(obj, position, orientation)
            
            elif class_name == 'traffic_sign':
                self._add_traffic_sign(obj, position, orientation)
    
    def _add_traffic_light(self, detection, vehicle_pos, vehicle_ori):
        """Add traffic light detection"""
        # Estimate 3D position from 2D bounding box
        bbox = detection['bbox']
        center = detection['center']
        
        # Simple projection (assumes traffic light is at some distance in front)
        estimated_distance = 20.0  # meters (rough estimate)
        
        # Convert to global position
        angle = vehicle_ori[2]  # yaw
        light_x = vehicle_pos[0] + estimated_distance * np.cos(angle)
        light_y = vehicle_pos[1] + estimated_distance * np.sin(angle)
        light_z = 5.0  # Typical traffic light height
        
        position_3d = np.array([light_x, light_y, light_z])
        
        # Check if this is a known traffic light (merge nearby detections)
        merged = False
        for light_id, light_data in self.traffic_lights.items():
            dist = np.linalg.norm(position_3d - light_data['position'])
            if dist < self.merge_distance:
                # Update existing light (moving average)
                obs = light_data['observations']
                light_data['position'] = (light_data['position'] * obs + position_3d) / (obs + 1)
                light_data['observations'] += 1
                merged = True
                break
        
        if not merged:
            # New traffic light
            self.traffic_lights[self.next_light_id] = {
                'position': position_3d,
                'observations': 1,
                'state_history': [],
                'type': 'traffic_light'
            }
            self.next_light_id += 1
    
    def _add_traffic_sign(self, detection, vehicle_pos, vehicle_ori):
        """Add traffic sign detection"""
        # Similar to traffic light
        bbox = detection['bbox']
        center = detection['center']
        
        estimated_distance = 15.0
        angle = vehicle_ori[2]
        sign_x = vehicle_pos[0] + estimated_distance * np.cos(angle)
        sign_y = vehicle_pos[1] + estimated_distance * np.sin(angle)
        sign_z = 3.0  # Typical sign height
        
        position_3d = np.array([sign_x, sign_y, sign_z])
        
        # Check if known sign
        merged = False
        for sign_id, sign_data in self.traffic_signs.items():
            dist = np.linalg.norm(position_3d - sign_data['position'])
            if dist < self.merge_distance:
                obs = sign_data['observations']
                sign_data['position'] = (sign_data['position'] * obs + position_3d) / (obs + 1)
                sign_data['observations'] += 1
                merged = True
                break
        
        if not merged:
            self.traffic_signs[self.next_sign_id] = {
                'position': position_3d,
                'observations': 1,
                'type': 'traffic_sign'
            }
            self.next_sign_id += 1
    
    def get_traffic_lights(self):
        """Get all mapped traffic lights"""
        return self.traffic_lights
    
    def get_traffic_signs(self):
        """Get all mapped traffic signs"""
        return self.traffic_signs
    
    def save_traffic_elements(self, filepath):
        """Save traffic elements to JSON"""
        data = {
            'traffic_lights': {
                str(light_id): {
                    'position': light_data['position'].tolist(),
                    'observations': light_data['observations'],
                    'type': light_data['type']
                }
                for light_id, light_data in self.traffic_lights.items()
            },
            'traffic_signs': {
                str(sign_id): {
                    'position': sign_data['position'].tolist(),
                    'observations': sign_data['observations'],
                    'type': sign_data['type']
                }
                for sign_id, sign_data in self.traffic_signs.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {len(self.traffic_lights)} traffic lights and {len(self.traffic_signs)} traffic signs")


class HDMapBuilder:
    """
    Complete HD Map Builder - integrates all components
    """
    
    def __init__(self, output_dir='./hd_maps'):
        """Initialize HD Map Builder"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.point_cloud_builder = PointCloudMapBuilder(voxel_size=0.2)
        self.lane_extractor = LaneExtractor()
        self.traffic_mapper = TrafficElementMapper(merge_distance=5.0)
        
        # Statistics
        self.frames_processed = 0
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("HD MAP BUILDER INITIALIZED")
        print("="*70)
        print(f"Output directory: {output_dir}")
        print("="*70 + "\n")
    
    def process_sensor_data(self, sensor_package):
        """
        Process incoming sensor data package
        
        Args:
            sensor_package: Dict containing sensor data from InterFuser agent
        """
        # Extract data
        vehicle_state = sensor_package.get('vehicle_state', {})
        position = vehicle_state.get('position', [0, 0, 0])
        compass = vehicle_state.get('compass', 0)
        orientation = [0, 0, compass]  # [roll, pitch, yaw]
        
        # Process LiDAR
        if 'lidar' in sensor_package:
            lidar_points = sensor_package['lidar']
            self.point_cloud_builder.add_lidar_scan(lidar_points, position, orientation)
        
        # Process images for lane extraction
        if 'images' in sensor_package and 'segmentation' in sensor_package:
            # Decompress and process
            pass  # Will be implemented with full integration
        
        # Process tracked objects for traffic elements
        tracked_objects = sensor_package.get('tracking', {}).get('tracked_objects', {})
        if tracked_objects:
            seg_image = None  # Need to get from package
            self.traffic_mapper.process_frame(seg_image, tracked_objects, position, orientation)
        
        self.frames_processed += 1
        
        # Periodic status update
        if self.frames_processed % 100 == 0:
            self._print_status()
    
    def _print_status(self):
        """Print current building status"""
        elapsed = time.time() - self.start_time
        fps = self.frames_processed / elapsed if elapsed > 0 else 0
        
        pc_stats = self.point_cloud_builder.get_statistics()
        
        print(f"\n{'='*70}")
        print(f"HD MAP BUILDING STATUS - Frame {self.frames_processed}")
        print(f"{'='*70}")
        print(f"Time elapsed: {elapsed:.1f}s | Processing rate: {fps:.1f} FPS")
        print(f"Point Cloud: {pc_stats['current_voxels']} voxels | {pc_stats['total_scans_processed']} scans")
        print(f"Lanes: {len(self.lane_extractor.get_all_lanes())} segments")
        print(f"Traffic Lights: {len(self.traffic_mapper.get_traffic_lights())}")
        print(f"Traffic Signs: {len(self.traffic_mapper.get_traffic_signs())}")
        print(f"{'='*70}\n")
    
    def save_hd_map(self, map_name='hd_map'):
        """Save complete HD map to disk"""
        print(f"\nSaving HD Map: {map_name}...")
        
        # Save point cloud
        pc_path = os.path.join(self.output_dir, f'{map_name}_pointcloud.pcd')
        self.point_cloud_builder.save_point_cloud(pc_path)
        
        # Save lanes
        lanes_path = os.path.join(self.output_dir, f'{map_name}_lanes.json')
        self.lane_extractor.save_lanes(lanes_path)
        
        # Save traffic elements
        traffic_path = os.path.join(self.output_dir, f'{map_name}_traffic.json')
        self.traffic_mapper.save_traffic_elements(traffic_path)
        
        # Save combined map metadata
        metadata = {
            'map_name': map_name,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'frames_processed': self.frames_processed,
            'point_cloud_file': f'{map_name}_pointcloud.pcd',
            'lanes_file': f'{map_name}_lanes.json',
            'traffic_file': f'{map_name}_traffic.json',
            'statistics': {
                'point_cloud': self.point_cloud_builder.get_statistics(),
                'lanes': len(self.lane_extractor.get_all_lanes()),
                'traffic_lights': len(self.traffic_mapper.get_traffic_lights()),
                'traffic_signs': len(self.traffic_mapper.get_traffic_signs())
            }
        }
        
        metadata_path = os.path.join(self.output_dir, f'{map_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ HD Map saved successfully!")
        print(f"  Location: {self.output_dir}")
        print(f"  Files: {map_name}_pointcloud.pcd, {map_name}_lanes.json, {map_name}_traffic.json")
        print(f"  Metadata: {map_name}_metadata.json\n")
    
    def get_statistics(self):
        """Get current building statistics"""
        return {
            'frames_processed': self.frames_processed,
            'point_cloud': self.point_cloud_builder.get_statistics(),
            'lanes': len(self.lane_extractor.get_all_lanes()),
            'traffic_lights': len(self.traffic_mapper.get_traffic_lights()),
            'traffic_signs': len(self.traffic_mapper.get_traffic_signs())
        }


if __name__ == '__main__':
    # Test the HD Map Builder
    print("HD Map Builder - Test Mode")
    
    builder = HDMapBuilder(output_dir='./test_hd_maps')
    
    # Simulate some data
    for i in range(10):
        sensor_package = {
            'vehicle_state': {
                'position': [i * 5, 0, 2.5],
                'compass': 0
            },
            'lidar': np.random.randn(1000, 3) * 10,  # Random point cloud
            'tracking': {'tracked_objects': {}}
        }
        
        builder.process_sensor_data(sensor_package)
    
    builder.save_hd_map('test_map')
    print("\n✓ Test completed successfully!")
