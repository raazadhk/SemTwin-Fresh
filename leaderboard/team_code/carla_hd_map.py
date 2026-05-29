#!/usr/bin/env python3
"""
HD Map Module for InterFuser Agent
Extracts HD map from CARLA and visualizes it in Digital Twin
"""

import carla
import numpy as np
import cv2
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math


class CARLAHDMapExtractor:
    """
    Extracts HD map information from CARLA simulator
    Provides lane geometry, topology, traffic elements
    """
    def __init__(self, carla_world: carla.World):
        self.world = carla_world
        self.map = carla_world.get_map()
        
        # Cache for performance
        self.waypoints_cache = None
        self.topology_cache = None
        self.lane_markings_cache = {}
        
        print("✓ CARLA HD Map Extractor initialized")
        print(f"  Map name: {self.map.name}")
    
    def get_local_lanes(self, vehicle_location: carla.Location, 
                       radius: float = 50.0) -> List[Dict]:
        """
        Get all lanes within radius of vehicle
        
        Args:
            vehicle_location: Vehicle position
            radius: Search radius in meters
        
        Returns:
            List of lane dictionaries with geometry and metadata
        """
        lanes = []
        
        # Get waypoints in radius
        waypoints = self.map.generate_waypoints(2.0)  # Every 2 meters
        
        for wp in waypoints:
            distance = wp.transform.location.distance(vehicle_location)
            
            if distance < radius:
                # Get lane information
                lane_id = wp.lane_id
                road_id = wp.road_id
                
                # Get lane geometry
                lane_width = wp.lane_width
                
                # Get lane type
                lane_type = wp.lane_type
                lane_type_str = self._lane_type_to_string(lane_type)
                
                # Get lane change permissions
                left_lane_change = wp.lane_change
                
                # Get successor waypoints (lane connectivity)
                next_wps = wp.next(2.0)
                successors = [w.lane_id for w in next_wps] if next_wps else []
                
                # Get lane markings
                left_marking = self._get_lane_marking_info(wp.left_lane_marking)
                right_marking = self._get_lane_marking_info(wp.right_lane_marking)
                
                lanes.append({
                    'waypoint': wp,
                    'lane_id': lane_id,
                    'road_id': road_id,
                    'position': (wp.transform.location.x, 
                               wp.transform.location.y,
                               wp.transform.location.z),
                    'width': lane_width,
                    'type': lane_type_str,
                    'successors': successors,
                    'left_marking': left_marking,
                    'right_marking': right_marking,
                    'distance': distance
                })
        
        return lanes
    
    def get_lane_boundaries(self, vehicle_location: carla.Location,
                           forward_distance: float = 50.0,
                           lateral_distance: float = 20.0) -> Dict:
        """
        Get lane boundary lines for visualization
        
        Args:
            vehicle_location: Vehicle position
            forward_distance: How far ahead to look (meters)
            lateral_distance: How far to sides (meters)
        
        Returns:
            Dictionary with left and right boundary points
        """
        # Get current waypoint
        current_wp = self.map.get_waypoint(vehicle_location)
        
        if not current_wp:
            return {'left': [], 'right': [], 'center': []}
        
        boundaries = {
            'left': [],
            'right': [],
            'center': [],
            'left_marking_type': self._get_lane_marking_info(current_wp.left_lane_marking),
            'right_marking_type': self._get_lane_marking_info(current_wp.right_lane_marking)
        }
        
        # Traverse forward along the lane
        next_wps = [current_wp]
        distance = 0.0
        step = 2.0  # Sample every 2 meters
        
        while distance < forward_distance and next_wps:
            wp = next_wps[0]
            
            # Center line
            center_loc = wp.transform.location
            boundaries['center'].append((center_loc.x, center_loc.y, center_loc.z))
            
            # Calculate left and right boundary points
            # Get perpendicular direction (left)
            forward = wp.transform.get_forward_vector()
            right = carla.Vector3D(forward.y, -forward.x, 0)
            
            # Left boundary
            left_offset = wp.lane_width / 2.0
            left_loc = center_loc + carla.Location(
                x=right.x * -left_offset,
                y=right.y * -left_offset,
                z=0
            )
            boundaries['left'].append((left_loc.x, left_loc.y, left_loc.z))
            
            # Right boundary
            right_offset = wp.lane_width / 2.0
            right_loc = center_loc + carla.Location(
                x=right.x * right_offset,
                y=right.y * right_offset,
                z=0
            )
            boundaries['right'].append((right_loc.x, right_loc.y, right_loc.z))
            
            # Get next waypoints
            next_wps = wp.next(step)
            distance += step
        
        return boundaries
    
    def get_traffic_lights(self, vehicle_location: carla.Location,
                          radius: float = 50.0) -> List[Dict]:
        """
        Get traffic lights within radius
        
        Args:
            vehicle_location: Vehicle position
            radius: Search radius
        
        Returns:
            List of traffic light dictionaries
        """
        traffic_lights = []
        
        # Get all traffic lights in the world
        actors = self.world.get_actors().filter('traffic.traffic_light*')
        
        for actor in actors:
            distance = actor.get_location().distance(vehicle_location)
            
            if distance < radius:
                # Get traffic light state
                state = actor.get_state()
                state_str = self._traffic_light_state_to_string(state)
                
                # Get affected waypoints
                affected_wps = actor.get_affected_lane_waypoints()
                affected_lanes = [wp.lane_id for wp in affected_wps] if affected_wps else []
                
                traffic_lights.append({
                    'id': actor.id,
                    'location': actor.get_location(),
                    'position': (actor.get_location().x,
                               actor.get_location().y,
                               actor.get_location().z),
                    'state': state_str,
                    'affected_lanes': affected_lanes,
                    'distance': distance,
                    'actor': actor
                })
        
        return traffic_lights
    
    def get_traffic_signs(self, vehicle_location: carla.Location,
                         radius: float = 50.0) -> List[Dict]:
        """
        Get traffic signs (stop signs, speed limits, etc.)
        
        Args:
            vehicle_location: Vehicle position
            radius: Search radius
        
        Returns:
            List of traffic sign dictionaries
        """
        traffic_signs = []
        
        # Get stop signs
        actors = self.world.get_actors().filter('traffic.stop*')
        for actor in actors:
            distance = actor.get_location().distance(vehicle_location)
            if distance < radius:
                traffic_signs.append({
                    'type': 'stop',
                    'location': actor.get_location(),
                    'position': (actor.get_location().x,
                               actor.get_location().y,
                               actor.get_location().z),
                    'distance': distance
                })
        
        # Get speed limit signs
        actors = self.world.get_actors().filter('traffic.speed_limit*')
        for actor in actors:
            distance = actor.get_location().distance(vehicle_location)
            if distance < radius:
                # Extract speed limit from actor type_id
                type_id = actor.type_id
                speed_limit = self._extract_speed_limit(type_id)
                
                traffic_signs.append({
                    'type': 'speed_limit',
                    'value': speed_limit,
                    'location': actor.get_location(),
                    'position': (actor.get_location().x,
                               actor.get_location().y,
                               actor.get_location().z),
                    'distance': distance
                })
        
        # Get yield signs
        actors = self.world.get_actors().filter('traffic.yield*')
        for actor in actors:
            distance = actor.get_location().distance(vehicle_location)
            if distance < radius:
                traffic_signs.append({
                    'type': 'yield',
                    'location': actor.get_location(),
                    'position': (actor.get_location().x,
                               actor.get_location().y,
                               actor.get_location().z),
                    'distance': distance
                })
        
        return traffic_signs
    
    def get_current_lane_info(self, vehicle_location: carla.Location,
                              vehicle_rotation: carla.Rotation) -> Optional[Dict]:
        """
        Get detailed information about vehicle's current lane
        
        Args:
            vehicle_location: Vehicle position
            vehicle_rotation: Vehicle orientation
        
        Returns:
            Dictionary with current lane information
        """
        waypoint = self.map.get_waypoint(vehicle_location)
        
        if not waypoint:
            return None
        
        # Get speed limit (if available)
        # CARLA doesn't directly provide speed limits, estimate from map
        # In real HD maps, this would be explicitly stored
        is_highway = waypoint.lane_type == carla.LaneType.Driving and waypoint.lane_width > 3.0
        estimated_speed_limit = 90 if is_highway else 50  # km/h
        
        return {
            'lane_id': waypoint.lane_id,
            'road_id': waypoint.road_id,
            'section_id': waypoint.section_id,
            'width': waypoint.lane_width,
            'type': self._lane_type_to_string(waypoint.lane_type),
            'speed_limit': estimated_speed_limit,
            'is_junction': waypoint.is_junction,
            'left_marking': self._get_lane_marking_info(waypoint.left_lane_marking),
            'right_marking': self._get_lane_marking_info(waypoint.right_lane_marking),
            'lane_change': self._lane_change_to_string(waypoint.lane_change)
        }
    
    def _lane_type_to_string(self, lane_type) -> str:
        """Convert CARLA lane type to string"""
        type_map = {
            carla.LaneType.Driving: 'driving',
            carla.LaneType.Parking: 'parking',
            carla.LaneType.Bidirectional: 'bidirectional',
            carla.LaneType.Biking: 'biking',
            carla.LaneType.Sidewalk: 'sidewalk',
            carla.LaneType.Shoulder: 'shoulder',
            carla.LaneType.Stop: 'stop',
            carla.LaneType.NONE: 'none',
            carla.LaneType.Any: 'any'
        }
        return type_map.get(lane_type, 'unknown')
    
    def _get_lane_marking_info(self, marking) -> Dict:
        """Get lane marking information"""
        if not marking:
            return {'type': 'none', 'color': 'white'}
        
        # Marking type
        type_map = {
            carla.LaneMarkingType.Solid: 'solid',
            carla.LaneMarkingType.Broken: 'broken',
            carla.LaneMarkingType.SolidSolid: 'double_solid',
            carla.LaneMarkingType.SolidBroken: 'solid_broken',
            carla.LaneMarkingType.BrokenSolid: 'broken_solid',
            carla.LaneMarkingType.BrokenBroken: 'double_broken',
            carla.LaneMarkingType.BottsDots: 'botts_dots',
            carla.LaneMarkingType.Grass: 'grass',
            carla.LaneMarkingType.Curb: 'curb',
            carla.LaneMarkingType.NONE: 'none'
        }
        marking_type = type_map.get(marking.type, 'unknown')
        
        # Marking color
        color_map = {
            carla.LaneMarkingColor.White: 'white',
            carla.LaneMarkingColor.Blue: 'blue',
            carla.LaneMarkingColor.Green: 'green',
            carla.LaneMarkingColor.Red: 'red',
            carla.LaneMarkingColor.Yellow: 'yellow',
        }
        marking_color = color_map.get(marking.color, 'white')
        
        return {
            'type': marking_type,
            'color': marking_color,
            'width': marking.width if hasattr(marking, 'width') else 0.15
        }
    
    def _traffic_light_state_to_string(self, state) -> str:
        """Convert traffic light state to string"""
        state_map = {
            carla.TrafficLightState.Red: 'red',
            carla.TrafficLightState.Yellow: 'yellow',
            carla.TrafficLightState.Green: 'green',
            carla.TrafficLightState.Off: 'off',
            carla.TrafficLightState.Unknown: 'unknown'
        }
        return state_map.get(state, 'unknown')
    
    def _lane_change_to_string(self, lane_change) -> str:
        """Convert lane change enum to string"""
        change_map = {
            carla.LaneChange.NONE: 'none',
            carla.LaneChange.Right: 'right',
            carla.LaneChange.Left: 'left',
            carla.LaneChange.Both: 'both'
        }
        return change_map.get(lane_change, 'none')
    
    def _extract_speed_limit(self, type_id: str) -> int:
        """Extract speed limit from CARLA actor type_id"""
        # Example: "traffic.speed_limit.30" -> 30
        parts = type_id.split('.')
        if len(parts) >= 3:
            try:
                return int(parts[-1])
            except:
                pass
        return 50  # Default


class HDMapVisualizer:
    """
    Visualizes HD map on camera frames
    Projects 3D map elements onto 2D image
    """
    def __init__(self, camera_params: Dict):
        """
        Initialize with camera intrinsic parameters
        
        Args:
            camera_params: Dictionary with:
                - fov: Field of view in degrees
                - width: Image width
                - height: Image height
        """
        self.fov = camera_params.get('fov', 90.0)
        self.width = camera_params['width']
        self.height = camera_params['height']
        
        # Camera intrinsic matrix
        self.K = self._build_projection_matrix()
        
        print("✓ HD Map Visualizer initialized")
        print(f"  Camera: {self.width}×{self.height}, FOV={self.fov}°")
    
    def _build_projection_matrix(self) -> np.ndarray:
        """Build camera intrinsic matrix K"""
        focal = self.width / (2.0 * np.tan(self.fov * np.pi / 360.0))
        K = np.array([
            [focal, 0, self.width / 2.0],
            [0, focal, self.height / 2.0],
            [0, 0, 1.0]
        ])
        return K
    
    def world_to_camera(self, world_point: Tuple[float, float, float],
                       camera_transform: carla.Transform) -> np.ndarray:
        """
        Transform world coordinates to camera coordinates
        
        Args:
            world_point: (x, y, z) in world frame
            camera_transform: Camera pose
        
        Returns:
            (x, y, z) in camera frame
        """
        # World point
        p_world = np.array([world_point[0], world_point[1], world_point[2], 1.0])
        
        # Camera pose (world to camera)
        cam_loc = camera_transform.location
        cam_rot = camera_transform.rotation
        
        # Build transformation matrix (world -> camera)
        # CARLA uses left-handed coordinate system
        yaw = np.deg2rad(cam_rot.yaw)
        pitch = np.deg2rad(cam_rot.pitch)
        roll = np.deg2rad(cam_rot.roll)
        
        # Rotation matrix
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R = R_yaw @ R_pitch @ R_roll
        
        # Translation
        t = np.array([cam_loc.x, cam_loc.y, cam_loc.z])
        
        # Transform to camera frame
        p_cam_xyz = R.T @ (p_world[:3] - t)
        
        # CARLA camera: X=right, Y=forward, Z=up
        # Convert to standard camera: X=right, Y=down, Z=forward
        p_cam = np.array([p_cam_xyz[1], -p_cam_xyz[2], p_cam_xyz[0]])
        
        return p_cam
    
    def camera_to_image(self, camera_point: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Project camera coordinates to image pixels
        
        Args:
            camera_point: (x, y, z) in camera frame
        
        Returns:
            (u, v) pixel coordinates or None if behind camera
        """
        # Check if point is in front of camera
        if camera_point[2] <= 0:
            return None
        
        # Project to image plane
        p_2d = self.K @ camera_point
        u = int(p_2d[0] / p_2d[2])
        v = int(p_2d[1] / p_2d[2])
        
        # Check if in image bounds
        if 0 <= u < self.width and 0 <= v < self.height:
            return (u, v)
        
        return None
    
    def project_world_to_image(self, world_point: Tuple[float, float, float],
                              camera_transform: carla.Transform) -> Optional[Tuple[int, int]]:
        """
        Complete pipeline: world -> camera -> image
        
        Args:
            world_point: (x, y, z) in world coordinates
            camera_transform: Camera pose
        
        Returns:
            (u, v) pixel coordinates or None if not visible
        """
        # Transform to camera frame
        p_cam = self.world_to_camera(world_point, camera_transform)
        
        # Project to image
        return self.camera_to_image(p_cam)
    
    def draw_lane_boundaries(self, image: np.ndarray,
                            boundaries: Dict,
                            camera_transform: carla.Transform,
                            color_left: Tuple[int, int, int] = (255, 255, 0),
                            color_right: Tuple[int, int, int] = (255, 255, 0),
                            color_center: Tuple[int, int, int] = (0, 255, 255),
                            thickness: int = 2) -> np.ndarray:
        """
        Draw lane boundaries on image
        
        Args:
            image: Input image (will be modified)
            boundaries: Dictionary with left, right, center points
            camera_transform: Camera pose
            color_left: BGR color for left boundary
            color_right: BGR color for right boundary
            color_center: BGR color for center line
            thickness: Line thickness
        
        Returns:
            Modified image
        """
        img = image.copy()
        
        # Draw left boundary
        if boundaries['left']:
            self._draw_polyline(img, boundaries['left'], camera_transform,
                              color_left, thickness,
                              line_type=boundaries.get('left_marking_type', {}).get('type', 'solid'))
        
        # Draw right boundary
        if boundaries['right']:
            self._draw_polyline(img, boundaries['right'], camera_transform,
                              color_right, thickness,
                              line_type=boundaries.get('right_marking_type', {}).get('type', 'solid'))
        
        # Draw center line (dashed, thinner)
        if boundaries['center']:
            self._draw_polyline(img, boundaries['center'], camera_transform,
                              color_center, max(1, thickness - 1),
                              line_type='broken')
        
        return img
    
    def _draw_polyline(self, image: np.ndarray, points_3d: List[Tuple],
                      camera_transform: carla.Transform,
                      color: Tuple[int, int, int], thickness: int,
                      line_type: str = 'solid'):
        """Draw a polyline on image"""
        # Project all points
        points_2d = []
        for point_3d in points_3d:
            pixel = self.project_world_to_image(point_3d, camera_transform)
            if pixel:
                points_2d.append(pixel)
        
        # Draw lines
        if len(points_2d) < 2:
            return
        
        if line_type == 'solid':
            # Solid line
            for i in range(len(points_2d) - 1):
                cv2.line(image, points_2d[i], points_2d[i + 1], color, thickness)
        else:
            # Dashed/broken line
            for i in range(0, len(points_2d) - 1, 2):
                cv2.line(image, points_2d[i], points_2d[i + 1], color, thickness)
    
    def draw_traffic_light(self, image: np.ndarray,
                          traffic_light: Dict,
                          camera_transform: carla.Transform) -> np.ndarray:
        """
        Draw traffic light indicator on image
        
        Args:
            image: Input image
            traffic_light: Traffic light dictionary
            camera_transform: Camera pose
        
        Returns:
            Modified image
        """
        img = image.copy()
        
        # Project traffic light position
        pixel = self.project_world_to_image(
            traffic_light['position'],
            camera_transform
        )
        
        if pixel:
            # Color based on state
            state_colors = {
                'red': (0, 0, 255),
                'yellow': (0, 255, 255),
                'green': (0, 255, 0),
                'off': (100, 100, 100),
                'unknown': (200, 200, 200)
            }
            color = state_colors.get(traffic_light['state'], (200, 200, 200))
            
            # Draw circle
            cv2.circle(img, pixel, 12, color, -1)
            cv2.circle(img, pixel, 12, (0, 0, 0), 2)
            
            # Draw state text
            state_text = traffic_light['state'].upper()
            cv2.putText(img, state_text, (pixel[0] + 20, pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img
    
    def draw_traffic_sign(self, image: np.ndarray,
                         sign: Dict,
                         camera_transform: carla.Transform) -> np.ndarray:
        """
        Draw traffic sign on image
        
        Args:
            image: Input image
            sign: Sign dictionary
            camera_transform: Camera pose
        
        Returns:
            Modified image
        """
        img = image.copy()
        
        # Project sign position
        pixel = self.project_world_to_image(sign['position'], camera_transform)
        
        if pixel:
            if sign['type'] == 'stop':
                # Draw red octagon
                cv2.circle(img, pixel, 15, (0, 0, 255), -1)
                cv2.circle(img, pixel, 15, (255, 255, 255), 2)
                cv2.putText(img, "STOP", (pixel[0] - 20, pixel[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            elif sign['type'] == 'speed_limit':
                # Draw speed limit sign
                cv2.circle(img, pixel, 15, (255, 255, 255), -1)
                cv2.circle(img, pixel, 15, (0, 0, 255), 2)
                speed_text = str(int(sign.get('value', 50)))
                cv2.putText(img, speed_text, (pixel[0] - 10, pixel[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            elif sign['type'] == 'yield':
                # Draw yellow triangle
                pts = np.array([
                    [pixel[0], pixel[1] - 15],
                    [pixel[0] - 13, pixel[1] + 10],
                    [pixel[0] + 13, pixel[1] + 10]
                ], np.int32)
                cv2.fillPoly(img, [pts], (0, 255, 255))
                cv2.polylines(img, [pts], True, (0, 0, 0), 2)
        
        return img
    
    def draw_lane_info(self, image: np.ndarray,
                      lane_info: Dict,
                      position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Draw current lane information as text overlay
        
        Args:
            image: Input image
            lane_info: Current lane information
            position: Text position (x, y)
        
        Returns:
            Modified image
        """
        img = image.copy()
        
        if not lane_info:
            return img
        
        # Create semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (position[0] - 5, position[1] - 20),
                     (position[0] + 350, position[1] + 140),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Draw text
        y_offset = position[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255)
        
        texts = [
            f"CURRENT LANE: {lane_info['lane_id']} (Road {lane_info['road_id']})",
            f"Type: {lane_info['type']} | Width: {lane_info['width']:.1f}m",
            f"Speed Limit: {lane_info['speed_limit']} km/h",
            f"Lane Change: {lane_info['lane_change']}",
            f"Markings: L={lane_info['left_marking']['type']} | R={lane_info['right_marking']['type']}",
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(img, text, (position[0], y_offset + i * 25),
                       font, 0.5, color, 1, cv2.LINE_AA)
        
        return img


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HD MAP MODULE FOR INTERFUSER AGENT")
    print("="*70)
    print("\nThis module provides:")
    print("  ✓ HD map extraction from CARLA")
    print("  ✓ Lane geometry and boundaries")
    print("  ✓ Traffic light detection and state")
    print("  ✓ Traffic sign detection")
    print("  ✓ Real-time visualization on camera frames")
    print("\nIntegrate with InterFuser agent to visualize HD maps!")
    print("="*70 + "\n")
