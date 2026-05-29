"""
Simple HD Map Generator - Guaranteed to Show Something!
Works directly with InterfuserAgent data without complex processing
"""

import numpy as np
import cv2


def create_simple_hd_map(traffic_meta, ego_speed=0, waypoints=None, map_size=800):
    """
    Create a simple but visible HD map from traffic_meta
    
    Args:
        traffic_meta: 20x20x7 or 400x7 array from InterfuserAgent
        ego_speed: Vehicle speed in m/s
        waypoints: Predicted waypoints
        map_size: Output size in pixels
    
    Returns:
        RGB image of HD map
    """
    
    # Create blank map
    hd_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    
    # Dark background with blue tint
    hd_map[:] = (30, 25, 20)
    
    # Reshape traffic_meta if needed
    if len(traffic_meta.shape) == 2 and traffic_meta.shape == (400, 7):
        traffic_meta = traffic_meta.reshape(20, 20, 7)
    
    center = map_size // 2
    pixels_per_meter = 20  # 20x20 grid covers 40x40 meters
    
    # Draw reference grid (every 10 meters)
    grid_color = (40, 35, 30)
    for i in range(0, map_size, pixels_per_meter * 10):
        cv2.line(hd_map, (i, 0), (i, map_size), grid_color, 1)
        cv2.line(hd_map, (0, i), (map_size, i), grid_color, 1)
    
    # Draw center cross (brighter)
    cv2.line(hd_map, (center, 0), (center, map_size), (50, 50, 50), 1)
    cv2.line(hd_map, (0, center), (map_size, center), (50, 50, 50), 1)
    
    # Draw road area (approximate circle around ego)
    cv2.circle(hd_map, (center, center), int(30 * pixels_per_meter), (45, 45, 45), -1)
    
    # Draw detected objects from traffic_meta
    for i in range(20):
        for j in range(20):
            cell = traffic_meta[i, j]
            
            # Check if there's an object
            vehicle_prob = cell[4]
            pedestrian_prob = cell[5]
            bike_prob = cell[6]
            
            max_prob = max(vehicle_prob, pedestrian_prob, bike_prob)
            
            if max_prob > 0.3:  # Confidence threshold
                # Calculate position
                grid_x = (j - 10) * 2.0  # Grid cell to meters
                grid_y = (10 - i) * 2.0
                
                # Add object offset
                obj_x = grid_x + cell[0]
                obj_y = grid_y + cell[1]
                
                # Convert to pixels
                px = int(center + obj_x * pixels_per_meter)
                py = int(center - obj_y * pixels_per_meter)
                
                # Check bounds
                if 0 <= px < map_size and 0 <= py < map_size:
                    # Determine object type and color
                    if vehicle_prob == max_prob:
                        color = (100, 100, 255)  # Red for vehicles
                        size = (18, 8)
                        label = 'V'
                    elif pedestrian_prob == max_prob:
                        color = (100, 255, 100)  # Green for pedestrians
                        size = (8, 8)
                        label = 'P'
                    else:  # bike
                        color = (255, 200, 100)  # Cyan for bikes
                        size = (12, 6)
                        label = 'B'
                    
                    # Draw object box
                    orientation = cell[3]
                    draw_oriented_box(hd_map, px, py, orientation, color, size)
                    
                    # Draw velocity arrow if moving
                    speed = cell[2]
                    if speed > 0.5:  # Moving
                        arrow_len = int(speed * 10)
                        end_x = px + int(np.cos(orientation) * arrow_len)
                        end_y = py - int(np.sin(orientation) * arrow_len)
                        cv2.arrowedLine(hd_map, (px, py), (end_x, end_y), 
                                      (255, 255, 0), 2, tipLength=0.3)
                    
                    # Draw label
                    cv2.putText(hd_map, label, (px-5, py-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw waypoints if available
    if waypoints is not None:
        if len(waypoints.shape) == 1:
            waypoints = waypoints.reshape(-1, 2)
        
        for i, wp in enumerate(waypoints):
            px = int(center + wp[0] * pixels_per_meter)
            py = int(center - wp[1] * pixels_per_meter)
            
            if 0 <= px < map_size and 0 <= py < map_size:
                # Color gradient: green -> yellow -> red
                ratio = i / len(waypoints)
                if ratio < 0.5:
                    color = (0, int(255 * (1 - ratio * 2)), int(255 * ratio * 2))
                else:
                    color = (0, int(255 * (1 - (ratio - 0.5) * 2)), 255)
                
                cv2.circle(hd_map, (px, py), 5, color, -1)
                
                # Connect waypoints
                if i > 0:
                    prev_wp = waypoints[i-1]
                    prev_px = int(center + prev_wp[0] * pixels_per_meter)
                    prev_py = int(center - prev_wp[1] * pixels_per_meter)
                    cv2.line(hd_map, (prev_px, prev_py), (px, py), color, 2)
    
    # Draw ego vehicle (always at center)
    ego_color = (100, 255, 100)  # Bright green
    draw_oriented_box(hd_map, center, center, 0, ego_color, (25, 12))
    
    # Draw heading indicator
    cv2.circle(hd_map, (center, center - 12), 5, (255, 255, 255), -1)
    
    # Add info text
    cv2.putText(hd_map, f'Speed: {ego_speed*3.6:.1f} km/h', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(hd_map, 'Scale: 10m grid', (10, map_size - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return hd_map


def draw_oriented_box(img, x, y, yaw, color, size):
    """Draw an oriented rectangle"""
    length, width = size
    
    # Create corners
    corners = np.array([
        [-width/2, -length/2],
        [width/2, -length/2],
        [width/2, length/2],
        [-width/2, length/2]
    ])
    
    # Rotate
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    rotated = corners @ rotation.T
    
    # Translate
    rotated[:, 0] += x
    rotated[:, 1] += y
    
    # Draw
    pts = rotated.astype(np.int32)
    cv2.fillPoly(img, [pts], tuple(c // 2 for c in color))  # Shadow
    cv2.fillPoly(img, [pts], color)  # Fill
    cv2.polylines(img, [pts], True, (255, 255, 255), 1)  # Border


# Test function
if __name__ == '__main__':
    # Test with dummy data
    traffic_meta = np.random.rand(20, 20, 7)
    traffic_meta[:, :, 4:7] = 0  # Clear probabilities
    
    # Add a few test objects
    traffic_meta[10, 12, 4] = 0.9  # Vehicle ahead
    traffic_meta[8, 10, 5] = 0.8   # Pedestrian
    traffic_meta[10, 8, 6] = 0.7   # Bike
    
    # Generate map
    hd_map = create_simple_hd_map(traffic_meta, ego_speed=10, waypoints=None)
    
    # Save
    cv2.imwrite('test_hd_map.png', hd_map)
    print("Test map saved to test_hd_map.png")
