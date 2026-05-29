"""
HD Map Builder for InterfuserAgent
Extracts proper lane geometry, markings, and traffic elements from CARLA
Creates Tesla FSD-style HD map visualization
"""

import numpy as np
import cv2
import carla
from typing import List, Dict, Tuple, Optional
import math


class HDMapBuilder:
    """
    Builds HD maps from CARLA world data
    Extracts lane-level geometry, markings, traffic signs, and creates visualization
    """
    
    def __init__(self, carla_world: carla.World, pixels_per_meter: float = 5.0):
        self.world = carla_world
        self.map = carla_world.get_map()
        self.pixels_per_meter = pixels_per_meter
        
        # Cache for performance
        self.cached_waypoints = None
        self.cached_topology = None
        
    def build_local_hd_map(self, ego_location: carla.Location, 
                          ego_rotation: float,
                          radius: float = 50.0,
                          map_size: Tuple[int, int] = (800, 800)) -> Tuple[np.ndarray, Dict]:
        """
        Build HD map around ego vehicle
        
        Args:
            ego_location: Ego vehicle location
            ego_rotation: Ego vehicle yaw in radians
            radius: Radius around ego vehicle in meters
            map_size: Output map size in pixels
            
        Returns:
            map_image: RGB image of HD map
            map_data: Dictionary with structured map data
        """
        
        # Create blank map
        map_img = np.zeros((map_size[1], map_size[0], 3), dtype=np.uint8)
        
        # Dark background
        map_img[:] = (20, 20, 20)
        
        # Get all map elements
        lane_data = self._get_lane_geometry(ego_location, radius)
        lane_markings = self._get_lane_markings(ego_location, radius)
        traffic_elements = self._get_traffic_elements(ego_location, radius)
        
        # Transform to ego-centric coordinates
        def world_to_map(x, y):
            """Convert world coordinates to map pixel coordinates"""
            # Translate to ego-centric
            dx = x - ego_location.x
            dy = y - ego_location.y
            
            # Rotate to ego heading
            cos_r = np.cos(-ego_rotation)
            sin_r = np.sin(-ego_rotation)
            rx = dx * cos_r - dy * sin_r
            ry = dx * sin_r + dy * cos_r
            
            # Convert to pixels (ego is at center)
            px = int(map_size[0] / 2 + rx * self.pixels_per_meter)
            py = int(map_size[1] / 2 - ry * self.pixels_per_meter)  # Flip y
            
            return px, py
        
        # Draw road surface (darker gray)
        for lane in lane_data:
            points = []
            for wp in lane['waypoints']:
                px, py = world_to_map(wp['x'], wp['y'])
                if 0 <= px < map_size[0] and 0 <= py < map_size[1]:
                    points.append([px, py])
            
            if len(points) > 2:
                points = np.array(points, dtype=np.int32)
                # Draw road surface
                cv2.polylines(map_img, [points], False, (40, 40, 40), 
                            thickness=int(lane['width'] * self.pixels_per_meter))
        
        # Draw lane markings
        for marking in lane_markings:
            points = []
            for wp in marking['points']:
                px, py = world_to_map(wp[0], wp[1])
                if 0 <= px < map_size[0] and 0 <= py < map_size[1]:
                    points.append([px, py])
            
            if len(points) > 1:
                points = np.array(points, dtype=np.int32)
                
                # Color based on marking type
                if marking['color'] == 'Yellow':
                    color = (0, 255, 255)  # Yellow in BGR
                else:
                    color = (255, 255, 255)  # White
                
                # Thickness and style based on type
                if 'Solid' in marking['type']:
                    cv2.polylines(map_img, [points], False, color, thickness=2)
                elif 'Broken' in marking['type']:
                    # Draw dashed line
                    self._draw_dashed_line(map_img, points, color, dash_length=10)
        
        # Draw traffic elements
        for element in traffic_elements:
            px, py = world_to_map(element['x'], element['y'])
            if 0 <= px < map_size[0] and 0 <= py < map_size[1]:
                
                if element['type'] == 'traffic_light':
                    # Color based on state
                    if element['state'] == 'Red':
                        color = (0, 0, 255)
                    elif element['state'] == 'Yellow':
                        color = (0, 255, 255)
                    elif element['state'] == 'Green':
                        color = (0, 255, 0)
                    else:
                        color = (128, 128, 128)
                    
                    cv2.circle(map_img, (px, py), 8, color, -1)
                    cv2.circle(map_img, (px, py), 8, (255, 255, 255), 1)
                
                elif element['type'] == 'stop_sign':
                    # Red octagon
                    cv2.circle(map_img, (px, py), 10, (0, 0, 255), -1)
                    cv2.putText(map_img, 'STOP', (px-15, py+3),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw ego vehicle
        ego_px, ego_py = world_to_map(ego_location.x, ego_location.y)
        self._draw_vehicle(map_img, ego_px, ego_py, 0, (100, 255, 100))  # Green for ego
        
        # Compile map data
        map_data = {
            'lanes': lane_data,
            'markings': lane_markings,
            'traffic_elements': traffic_elements,
            'ego_location': {'x': ego_location.x, 'y': ego_location.y, 'z': ego_location.z},
            'ego_rotation': ego_rotation,
            'radius': radius
        }
        
        return map_img, map_data
    
    def _get_lane_geometry(self, location: carla.Location, radius: float) -> List[Dict]:
        """Extract lane geometry within radius"""
        lanes = []
        
        # Get waypoints at regular intervals
        waypoint_separation = 2.0  # meters
        waypoints = self.map.generate_waypoints(waypoint_separation)
        
        processed_lanes = set()
        
        for wp in waypoints:
            if wp.transform.location.distance(location) > radius:
                continue
            
            # Create unique lane ID
            lane_id = f"{wp.road_id}_{wp.section_id}_{wp.lane_id}"
            
            if lane_id in processed_lanes:
                continue
            processed_lanes.add(lane_id)
            
            # Get lane waypoints
            lane_waypoints = []
            current_wp = wp
            
            # Go forward
            for _ in range(int(radius / waypoint_separation)):
                if current_wp is None:
                    break
                if current_wp.transform.location.distance(location) > radius:
                    break
                
                lane_waypoints.append({
                    'x': current_wp.transform.location.x,
                    'y': current_wp.transform.location.y,
                    'z': current_wp.transform.location.z,
                    'yaw': current_wp.transform.rotation.yaw
                })
                
                next_wps = current_wp.next(waypoint_separation)
                current_wp = next_wps[0] if next_wps else None
            
            if len(lane_waypoints) > 2:
                lanes.append({
                    'lane_id': lane_id,
                    'road_id': wp.road_id,
                    'lane_type': str(wp.lane_type),
                    'width': wp.lane_width,
                    'waypoints': lane_waypoints
                })
        
        return lanes
    
    def _get_lane_markings(self, location: carla.Location, radius: float) -> List[Dict]:
        """Extract lane marking information"""
        markings = []
        
        waypoints = self.map.generate_waypoints(1.0)
        
        for wp in waypoints:
            if wp.transform.location.distance(location) > radius:
                continue
            
            # Left marking
            left_marking = wp.left_lane_marking
            if left_marking and left_marking.type != carla.LaneMarkingType.NONE:
                marking_type = str(left_marking.type).split('.')[-1]
                marking_color = str(left_marking.color).split('.')[-1]
                
                # Calculate marking position (offset from lane center)
                offset = wp.lane_width / 2
                marking_loc = wp.transform.location + carla.Location(
                    x=-offset * math.sin(math.radians(wp.transform.rotation.yaw)),
                    y=offset * math.cos(math.radians(wp.transform.rotation.yaw))
                )
                
                markings.append({
                    'type': marking_type,
                    'color': marking_color,
                    'side': 'left',
                    'points': [(marking_loc.x, marking_loc.y)]
                })
            
            # Right marking
            right_marking = wp.right_lane_marking
            if right_marking and right_marking.type != carla.LaneMarkingType.NONE:
                marking_type = str(right_marking.type).split('.')[-1]
                marking_color = str(right_marking.color).split('.')[-1]
                
                offset = wp.lane_width / 2
                marking_loc = wp.transform.location + carla.Location(
                    x=offset * math.sin(math.radians(wp.transform.rotation.yaw)),
                    y=-offset * math.cos(math.radians(wp.transform.rotation.yaw))
                )
                
                markings.append({
                    'type': marking_type,
                    'color': marking_color,
                    'side': 'right',
                    'points': [(marking_loc.x, marking_loc.y)]
                })
        
        # Group consecutive markings of same type
        grouped_markings = self._group_markings(markings)
        
        return grouped_markings
    
    def _group_markings(self, markings: List[Dict]) -> List[Dict]:
        """Group consecutive marking points"""
        if not markings:
            return []
        
        grouped = []
        current_group = {
            'type': markings[0]['type'],
            'color': markings[0]['color'],
            'side': markings[0]['side'],
            'points': [markings[0]['points'][0]]
        }
        
        for i in range(1, len(markings)):
            m = markings[i]
            if (m['type'] == current_group['type'] and 
                m['color'] == current_group['color'] and
                m['side'] == current_group['side']):
                current_group['points'].append(m['points'][0])
            else:
                if len(current_group['points']) > 1:
                    grouped.append(current_group)
                current_group = {
                    'type': m['type'],
                    'color': m['color'],
                    'side': m['side'],
                    'points': [m['points'][0]]
                }
        
        if len(current_group['points']) > 1:
            grouped.append(current_group)
        
        return grouped
    
    def _get_traffic_elements(self, location: carla.Location, radius: float) -> List[Dict]:
        """Get traffic lights and signs"""
        elements = []
        
        actors = self.world.get_actors()
        
        # Traffic lights
        for light in actors.filter('traffic.traffic_light*'):
            if light.get_location().distance(location) <= radius:
                elements.append({
                    'type': 'traffic_light',
                    'x': light.get_location().x,
                    'y': light.get_location().y,
                    'z': light.get_location().z,
                    'state': str(light.get_state()).split('.')[-1]
                })
        
        # Stop signs
        for sign in actors.filter('traffic.stop'):
            if sign.get_location().distance(location) <= radius:
                elements.append({
                    'type': 'stop_sign',
                    'x': sign.get_location().x,
                    'y': sign.get_location().y,
                    'z': sign.get_location().z
                })
        
        return elements
    
    def _draw_dashed_line(self, img: np.ndarray, points: np.ndarray, 
                         color: Tuple[int, int, int], dash_length: int = 10):
        """Draw dashed line"""
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            
            dist = np.linalg.norm(p2 - p1)
            if dist < 1:
                continue
            
            num_dashes = int(dist / dash_length)
            for j in range(num_dashes):
                if j % 2 == 0:  # Draw every other dash
                    t1 = j / num_dashes
                    t2 = min((j + 0.5) / num_dashes, 1.0)
                    
                    pt1 = (p1 + t1 * (p2 - p1)).astype(int)
                    pt2 = (p1 + t2 * (p2 - p1)).astype(int)
                    
                    cv2.line(img, tuple(pt1), tuple(pt2), color, 2)
    
    def _draw_vehicle(self, img: np.ndarray, x: int, y: int, 
                     yaw: float, color: Tuple[int, int, int]):
        """Draw vehicle as oriented rectangle"""
        # Vehicle dimensions in pixels
        length = 20
        width = 10
        
        # Create rectangle points
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
        
        # Draw
        cv2.fillPoly(img, [rotated_corners.astype(int)], color)
        cv2.polylines(img, [rotated_corners.astype(int)], True, (255, 255, 255), 1)
        
        # Draw heading indicator
        front_center = rotated_corners[1:3].mean(axis=0).astype(int)
        cv2.circle(img, tuple(front_center), 3, (255, 255, 255), -1)


def add_detected_objects_to_map(map_img: np.ndarray, 
                               objects: List[Dict],
                               ego_location: Tuple[float, float],
                               ego_rotation: float,
                               pixels_per_meter: float,
                               map_size: Tuple[int, int]) -> np.ndarray:
    """
    Add detected objects from InterfuserAgent to HD map
    
    Args:
        map_img: HD map image
        objects: List of detected objects from traffic_meta
        ego_location: Ego vehicle (x, y)
        ego_rotation: Ego yaw in radians
        pixels_per_meter: Scale factor
        map_size: Map dimensions
    """
    
    def world_to_map(x, y):
        dx = x - ego_location[0]
        dy = y - ego_location[1]
        
        cos_r = np.cos(-ego_rotation)
        sin_r = np.sin(-ego_rotation)
        rx = dx * cos_r - dy * sin_r
        ry = dx * sin_r + dy * cos_r
        
        px = int(map_size[0] / 2 + rx * pixels_per_meter)
        py = int(map_size[1] / 2 - ry * pixels_per_meter)
        
        return px, py
    
    for obj in objects:
        px, py = world_to_map(obj['position'][0], obj['position'][1])
        
        if 0 <= px < map_size[0] and 0 <= py < map_size[1]:
            # Color by class
            if obj['class'] == 'vehicle':
                color = (255, 100, 100)  # Light red
                size = (18, 8)
            elif obj['class'] == 'pedestrian':
                color = (100, 255, 100)  # Light green
                size = (8, 8)
            elif obj['class'] == 'bike':
                color = (255, 255, 100)  # Light yellow
                size = (12, 6)
            else:
                continue
            
            # Draw as oriented rectangle
            yaw = obj.get('orientation', 0)
            
            length, width = size
            corners = np.array([
                [-width/2, -length/2],
                [width/2, -length/2],
                [width/2, length/2],
                [-width/2, length/2]
            ])
            
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            rotated_corners = corners @ rotation_matrix.T
            
            rotated_corners[:, 0] += px
            rotated_corners[:, 1] += py
            
            cv2.fillPoly(map_img, [rotated_corners.astype(int)], color)
            cv2.polylines(map_img, [rotated_corners.astype(int)], True, (255, 255, 255), 1)
    
    return map_img
