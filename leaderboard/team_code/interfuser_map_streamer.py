"""
Real-time HD Map Streaming Server for InterfuserAgent
Streams BEV maps, sensor data, and predictions to remote visualization clients
"""

import socket
import struct
import pickle
import json
import time
import threading
import numpy as np
import cv2
import math
from collections import deque
from typing import Dict, Any, Optional
import base64

# Import enhanced renderer
try:
    from enhanced_bev_renderer import EnhancedBEVRenderer, LaneDetectorFromCamera
    ENHANCED_RENDERER_AVAILABLE = True
except ImportError:
    ENHANCED_RENDERER_AVAILABLE = False
    print("Warning: Enhanced BEV renderer not available")

# Import simple HD map generator (always works!)
try:
    from simple_hd_map import create_simple_hd_map
    SIMPLE_HD_MAP_AVAILABLE = True
except ImportError:
    SIMPLE_HD_MAP_AVAILABLE = False
    print("Warning: Simple HD map generator not available")

# Import real-time sensor-based HD map
try:
    from realtime_hd_map import RealTimeHDMapGenerator
    REALTIME_HD_MAP_AVAILABLE = True
except ImportError:
    REALTIME_HD_MAP_AVAILABLE = False
    print("Warning: Real-time HD map generator not available")

# Import vectorized HD map (clean Tesla-style)
try:
    from vectorized_hd_map import VectorizedHDMap
    VECTORIZED_HD_MAP_AVAILABLE = True
except ImportError:
    VECTORIZED_HD_MAP_AVAILABLE = False
    print("Warning: Vectorized HD map not available")


class InterfuserMapStreamer:
    """
    Streams HD map data from InterfuserAgent to remote clients.
    Supports multiple streaming protocols: TCP, WebSocket, and UDP broadcast.
    """
    
    def __init__(self, host='0.0.0.0', tcp_port=5555, udp_port=5556, max_clients=5):
        self.host = host
        self.tcp_port = tcp_port
        self.udp_port = udp_port
        self.max_clients = max_clients
        
        # TCP socket for reliable streaming
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # UDP socket for low-latency broadcast
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        self.running = False
        self.clients = []
        self.latest_data = None
        self.data_queue = deque(maxlen=30)  # Buffer for 1 second at 30 FPS
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
        # Enhanced rendering
        if ENHANCED_RENDERER_AVAILABLE:
            self.enhanced_renderer = EnhancedBEVRenderer()
            self.lane_detector = LaneDetectorFromCamera()
            print("✓ Enhanced BEV renderer initialized")
        else:
            self.enhanced_renderer = None
            self.lane_detector = None
        
        if SIMPLE_HD_MAP_AVAILABLE:
            print("✓ Simple HD map generator initialized (fallback)")
        else:
            print("Warning: Simple HD map generator not available")
        
        if REALTIME_HD_MAP_AVAILABLE:
            self.realtime_hd_map = RealTimeHDMapGenerator()
            print("✓ Real-time sensor HD map initialized")
        else:
            self.realtime_hd_map = None
        
        if VECTORIZED_HD_MAP_AVAILABLE:
            self.vectorized_hd_map = VectorizedHDMap()
            print("✓ Vectorized HD map initialized (Tesla-style)")
        else:
            self.vectorized_hd_map = None
        
    def start(self):
        """Start the streaming server"""
        try:
            self.tcp_socket.bind((self.host, self.tcp_port))
            self.tcp_socket.listen(self.max_clients)
            print(f"✓ TCP Map Streamer listening on {self.host}:{self.tcp_port}")
            
            self.udp_socket.bind((self.host, self.udp_port))
            print(f"✓ UDP Map Broadcaster ready on {self.host}:{self.udp_port}")
            
            self.running = True
            
            # Start client connection handler
            accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
            accept_thread.start()
            
            print("✓ InterfuserMapStreamer ready for connections")
            
        except Exception as e:
            print(f"✗ Failed to start streamer: {e}")
            raise
    
    def _accept_clients(self):
        """Accept incoming client connections"""
        while self.running:
            try:
                self.tcp_socket.settimeout(1.0)
                client_sock, client_addr = self.tcp_socket.accept()
                self.clients.append((client_sock, client_addr))
                print(f"✓ Client connected: {client_addr}")
                
                # Start thread to handle this client
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_sock, client_addr),
                    daemon=True
                )
                client_thread.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"✗ Error accepting client: {e}")
    
    def _handle_client(self, client_sock, client_addr):
        """Handle individual client connection"""
        try:
            while self.running:
                if self.latest_data is not None:
                    self._send_tcp_data(client_sock, self.latest_data)
                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"✗ Client {client_addr} disconnected: {e}")
        finally:
            client_sock.close()
            self.clients = [(s, a) for s, a in self.clients if a != client_addr]
    
    def extract_map_data(self, tick_data: Dict[str, Any], 
                        input_data: Dict[str, Any],
                        control: Any,
                        step: int,
                        carla_world=None,
                        ego_vehicle=None) -> Dict[str, Any]:
        """
        Extract all relevant map and sensor data from InterfuserAgent
        
        Args:
            tick_data: Data from agent.tick() containing sensors and processing results
            input_data: Processed input data fed to the model
            control: CARLA VehicleControl object
            step: Current step number
            
        Returns:
            Dictionary containing all map data for streaming
        """
        
        # Extract ego vehicle state
        ego_state = {
            'gps': tick_data['gps'].tolist() if isinstance(tick_data['gps'], np.ndarray) else list(tick_data['gps']),
            'compass': float(tick_data['compass']),
            'speed': float(tick_data['speed']),
            'position': tick_data['measurements'][:2],  # x, y from measurements
            'heading': float(tick_data['measurements'][2]),  # compass from measurements
        }
        
        # Extract traffic meta (20x20x7 grid with objects)
        traffic_meta = tick_data.get('raw', np.zeros((400, 7)))
        traffic_grid = traffic_meta.reshape(20, 20, 7)
        
        # Parse traffic objects from grid
        objects = self._parse_traffic_objects(traffic_grid)
        
        # Extract BEV feature map
        bev_feature = tick_data.get('bev_feature', None)
        if bev_feature is not None:
            bev_feature_encoded = base64.b64encode(bev_feature.tobytes()).decode('utf-8')
            bev_shape = bev_feature.shape
        else:
            bev_feature_encoded = None
            bev_shape = None
        
        # Generate HD map from CARLA (if available)
        hd_map_image = None
        hd_map_data = None
        
        if carla_world is not None and ego_vehicle is not None:
            try:
                from hd_map_builder import HDMapBuilder, add_detected_objects_to_map
                
                # Create HD map builder
                if not hasattr(self, 'hd_map_builder'):
                    self.hd_map_builder = HDMapBuilder(carla_world, pixels_per_meter=5.0)
                
                # Get ego location and rotation
                ego_transform = ego_vehicle.get_transform()
                ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
                
                # Build HD map
                hd_map_img, hd_map_struct = self.hd_map_builder.build_local_hd_map(
                    ego_location=ego_transform.location,
                    ego_rotation=ego_yaw_rad,
                    radius=50.0,
                    map_size=(800, 800)
                )
                
                # Add detected objects from InterfuserAgent
                hd_map_img = add_detected_objects_to_map(
                    hd_map_img,
                    objects,
                    ego_location=(ego_transform.location.x, ego_transform.location.y),
                    ego_rotation=ego_yaw_rad,
                    pixels_per_meter=5.0,
                    map_size=(800, 800)
                )
                
                # Encode HD map
                hd_map_image = self._encode_image(hd_map_img, quality=90)
                hd_map_data = hd_map_struct
                
            except Exception as e:
                print(f"HD map generation error: {e}")
                hd_map_image = None
                hd_map_data = None
        
        # Extract waypoints/trajectory
        # Note: pred_waypoints should be passed from run_step
        waypoints = tick_data.get('pred_waypoints', [])
        if isinstance(waypoints, np.ndarray):
            waypoints = waypoints.tolist()
        
        # Extract target point
        target_point = tick_data.get('target_point', [0, 0])
        if isinstance(target_point, np.ndarray):
            target_point = target_point.tolist()
        
        # Extract rendered maps (already RGB images)
        surround_map = tick_data.get('map', np.zeros((400, 400, 3)))
        map_t1 = tick_data.get('map_t1', np.zeros((200, 200, 3)))
        map_t2 = tick_data.get('map_t2', np.zeros((200, 200, 3)))
        
        # Generate HD map - Priority: Vectorized > Real-time > Simple
        enhanced_hd_map = None
        detected_lanes = None
        
        # Priority 1: Vectorized HD map (Clean Tesla-style!)
        if VECTORIZED_HD_MAP_AVAILABLE and self.vectorized_hd_map is not None:
            try:
                rgb_front = tick_data.get('rgb_raw', tick_data.get('rgb'))
                lidar_raw = tick_data.get('raw_lidar')
                
                pred_waypoints = tick_data.get('pred_waypoints')
                if pred_waypoints is not None and isinstance(pred_waypoints, np.ndarray):
                    if len(pred_waypoints.shape) == 1:
                        pred_waypoints = pred_waypoints.reshape(-1, 2)
                else:
                    pred_waypoints = None
                
                # Generate vectorized HD map
                enhanced_hd_map = self.vectorized_hd_map.generate_vectorized_map(
                    rgb_front=rgb_front,
                    lidar_data=lidar_raw,
                    traffic_meta=traffic_meta.reshape(20, 20, 7) if traffic_meta.shape == (400, 7) else traffic_meta,
                    ego_speed=ego_state['speed'],
                    waypoints=pred_waypoints,
                    map_size=800
                )
                
                enhanced_hd_map_encoded = self._encode_image(enhanced_hd_map, quality=90)
                if step % 50 == 0:
                    print(f"✓ Vectorized HD map generated (step {step})")
                
            except Exception as e:
                print(f"Vectorized HD map error: {e}")
                import traceback
                traceback.print_exc()
                enhanced_hd_map_encoded = None
        
        # Priority 2: Real-time sensor HD map
        elif REALTIME_HD_MAP_AVAILABLE and self.realtime_hd_map is not None:
            try:
                # Get sensor data
                rgb_front = tick_data.get('rgb_raw', tick_data.get('rgb'))
                lidar_raw = tick_data.get('raw_lidar')
                
                # Get predicted waypoints
                pred_waypoints = tick_data.get('pred_waypoints')
                if pred_waypoints is not None and isinstance(pred_waypoints, np.ndarray):
                    if len(pred_waypoints.shape) == 1:
                        pred_waypoints = pred_waypoints.reshape(-1, 2)
                else:
                    pred_waypoints = None
                
                # Generate real-time HD map from sensors
                enhanced_hd_map = self.realtime_hd_map.generate_hd_map(
                    rgb_front=rgb_front,
                    lidar_data=lidar_raw,
                    traffic_meta=traffic_meta.reshape(20, 20, 7) if traffic_meta.shape == (400, 7) else traffic_meta,
                    ego_speed=ego_state['speed'],
                    waypoints=pred_waypoints,
                    map_size=800
                )
                
                # Encode
                enhanced_hd_map_encoded = self._encode_image(enhanced_hd_map, quality=90)
                if step % 50 == 0:  # Print occasionally
                    print(f"✓ Real-time sensor HD map generated (step {step})")
                
            except Exception as e:
                print(f"Real-time HD map error: {e}")
                enhanced_hd_map_encoded = None
        
        # Priority 2: Try simple HD map as fallback
        elif SIMPLE_HD_MAP_AVAILABLE and traffic_meta is not None:
            try:
                # Get predicted waypoints
                pred_waypoints = tick_data.get('pred_waypoints')
                if pred_waypoints is not None and isinstance(pred_waypoints, np.ndarray):
                    if len(pred_waypoints.shape) == 1:
                        pred_waypoints = pred_waypoints.reshape(-1, 2)
                else:
                    pred_waypoints = None
                
                # Generate simple HD map (always works!)
                enhanced_hd_map = create_simple_hd_map(
                    traffic_meta=traffic_meta.reshape(20, 20, 7) if traffic_meta.shape == (400, 7) else traffic_meta,
                    ego_speed=ego_state['speed'],
                    waypoints=pred_waypoints,
                    map_size=800
                )
                
                # Encode
                enhanced_hd_map_encoded = self._encode_image(enhanced_hd_map, quality=90)
                
            except Exception as e:
                print(f"Simple HD map error: {e}")
                enhanced_hd_map_encoded = None
        
        # Priority 3: Try enhanced renderer as last resort
        elif self.enhanced_renderer is not None:
            try:
                # Get RGB front camera for lane detection
                rgb_front = tick_data.get('rgb_raw', tick_data.get('rgb'))
                
                # Detect lanes from camera
                if self.lane_detector is not None and rgb_front is not None:
                    detected_lanes = self.lane_detector.detect_lanes(rgb_front)
                
                # Get predicted waypoints
                pred_waypoints = tick_data.get('pred_waypoints')
                if pred_waypoints is not None and isinstance(pred_waypoints, np.ndarray):
                    if len(pred_waypoints.shape) == 1:
                        pred_waypoints = pred_waypoints.reshape(-1, 2)
                else:
                    pred_waypoints = None
                
                # Enhance the BEV map to look like HD map
                enhanced_hd_map = self.enhanced_renderer.enhance_bev_map(
                    bev_map=surround_map,
                    traffic_meta=traffic_meta.reshape(20, 20, 7) if traffic_meta.shape == (400, 7) else traffic_meta,
                    ego_speed=ego_state['speed'],
                    waypoints=pred_waypoints
                )
                
                # Encode enhanced map
                enhanced_hd_map_encoded = self._encode_image(enhanced_hd_map, quality=90)
                
            except Exception as e:
                print(f"Enhanced map generation error: {e}")
                enhanced_hd_map_encoded = None
        else:
            enhanced_hd_map_encoded = None
        
        # Encode original maps
        surround_map_encoded = self._encode_image(surround_map)
        map_t1_encoded = self._encode_image(map_t1)
        map_t2_encoded = self._encode_image(map_t2)
        
        # Extract LiDAR data
        raw_lidar = tick_data.get('raw_lidar', None)
        if raw_lidar is not None:
            # Subsample LiDAR for bandwidth (every 10th point)
            lidar_points = raw_lidar[::10, :3].tolist()
        else:
            lidar_points = []
        
        # Extract control commands
        control_data = {
            'throttle': float(control.throttle),
            'steer': float(control.steer),
            'brake': float(control.brake)
        }
        
        # Extract camera images (compressed)
        rgb_front = self._encode_image(tick_data.get('rgb_raw', tick_data.get('rgb')))
        rgb_left = self._encode_image(tick_data.get('rgb_left_raw', tick_data.get('rgb_left')))
        rgb_right = self._encode_image(tick_data.get('rgb_right_raw', tick_data.get('rgb_right')))
        
        # Compile complete map data
        map_data = {
            'timestamp': time.time(),
            'step': step,
            'ego_state': ego_state,
            'objects': objects,
            'waypoints': waypoints,
            'target_point': target_point,
            'control': control_data,
            'maps': {
                'bev_current': surround_map_encoded,
                'bev_t1': map_t1_encoded,
                'bev_t2': map_t2_encoded,
                'hd_map': enhanced_hd_map_encoded,  # Enhanced HD-style map
                'hd_map_carla': hd_map_image  # True HD map from CARLA (if available)
            },
            'detected_lanes': detected_lanes,  # Lane detection from camera
            'hd_map_data': hd_map_data,  # Structured HD map data from CARLA
            'bev_feature': {
                'data': bev_feature_encoded,
                'shape': bev_shape
            },
            'lidar': lidar_points,
            'cameras': {
                'front': rgb_front,
                'left': rgb_left,
                'right': rgb_right
            },
            'next_command': tick_data.get('next_command', 0)
        }
        
        return map_data
    
    def _parse_traffic_objects(self, traffic_grid: np.ndarray) -> list:
        """
        Parse traffic objects from 20x20x7 grid
        
        Grid channels (7):
        0-1: Object center position (dx, dy)
        2: Object speed
        3: Object orientation
        4-6: Object class probabilities (vehicle, pedestrian, bike)
        """
        objects = []
        
        for i in range(20):
            for j in range(20):
                cell = traffic_grid[i, j]
                
                # Check if there's a significant object (any class prob > 0.3)
                max_class_prob = np.max(cell[4:7])
                if max_class_prob > 0.3:
                    # Determine object class
                    class_idx = np.argmax(cell[4:7])
                    class_names = ['vehicle', 'pedestrian', 'bike']
                    
                    # Calculate world position (grid is 20x20 covering ~40m x 40m)
                    # Grid center is ego vehicle
                    grid_x = (j - 10) * 2.0  # meters
                    grid_y = (10 - i) * 2.0  # meters (flip y)
                    
                    # Add object offset
                    obj_x = grid_x + cell[0]
                    obj_y = grid_y + cell[1]
                    
                    obj = {
                        'class': class_names[class_idx],
                        'confidence': float(max_class_prob),
                        'position': [float(obj_x), float(obj_y)],
                        'velocity': float(cell[2]),
                        'orientation': float(cell[3]),
                        'grid_cell': [i, j]
                    }
                    objects.append(obj)
        
        return objects
    
    def _encode_image(self, img: np.ndarray, quality=85) -> str:
        """Encode image to JPEG base64 for efficient transmission"""
        if img is None or img.size == 0:
            return ""
        
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8')
    
    def stream_data(self, map_data: Dict[str, Any], protocol='tcp'):
        """
        Stream map data to connected clients
        
        Args:
            map_data: Dictionary containing all map information
            protocol: 'tcp' for reliable, 'udp' for low-latency broadcast, or 'both'
        """
        self.latest_data = map_data
        self.data_queue.append(map_data)
        self.frame_count += 1
        
        if protocol in ['udp', 'both']:
            self._broadcast_udp(map_data)
        
        # TCP is handled by client threads automatically via latest_data
        
        # Print statistics every 100 frames
        if self.frame_count % 100 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            print(f"📊 Streamed {self.frame_count} frames | {fps:.1f} FPS | {len(self.clients)} clients")
    
    def _send_tcp_data(self, client_sock: socket.socket, data: Dict[str, Any]):
        """Send data via TCP with length header"""
        try:
            # Serialize data
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            size = len(serialized)
            
            # Send size header (4 bytes) + data
            client_sock.sendall(struct.pack('!I', size))
            client_sock.sendall(serialized)
            
        except Exception as e:
            raise  # Let caller handle disconnection
    
    def _broadcast_udp(self, data: Dict[str, Any]):
        """Broadcast lightweight data via UDP"""
        try:
            # Create lightweight version for UDP (no images)
            lightweight = {
                'timestamp': data['timestamp'],
                'step': data['step'],
                'ego_state': data['ego_state'],
                'objects': data['objects'],
                'waypoints': data['waypoints'],
                'control': data['control']
            }
            
            # Serialize and send
            serialized = json.dumps(lightweight).encode('utf-8')
            
            # Split into chunks if needed (UDP max ~65KB)
            chunk_size = 60000
            if len(serialized) > chunk_size:
                # Send only essential data for UDP
                essential = {
                    'timestamp': data['timestamp'],
                    'ego_state': data['ego_state'],
                    'objects': data['objects'][:20],  # Limit objects
                    'waypoints': data['waypoints']
                }
                serialized = json.dumps(essential).encode('utf-8')
            
            self.udp_socket.sendto(serialized, ('<broadcast>', self.udp_port))
            
        except Exception as e:
            print(f"✗ UDP broadcast error: {e}")
    
    def stop(self):
        """Stop the streaming server"""
        print("Stopping InterfuserMapStreamer...")
        self.running = False
        
        # Close all client connections
        for client_sock, addr in self.clients:
            try:
                client_sock.close()
            except:
                pass
        
        # Close sockets
        try:
            self.tcp_socket.close()
            self.udp_socket.close()
        except:
            pass
        
        print("✓ InterfuserMapStreamer stopped")


# Helper function to integrate with InterfuserAgent
def integrate_with_agent():
    """
    Example integration code to add to InterfuserAgent.setup()
    """
    code = """
    # Add to InterfuserAgent.__init__ or setup():
    from interfuser_map_streamer import InterfuserMapStreamer
    
    self.map_streamer = InterfuserMapStreamer(
        host='0.0.0.0',
        tcp_port=5555,
        udp_port=5556
    )
    self.map_streamer.start()
    
    # Add to InterfuserAgent.run_step() right before 'return control':
    map_data = self.map_streamer.extract_map_data(tick_data, input_data, control, self.step)
    self.map_streamer.stream_data(map_data, protocol='both')
    
    # Add to InterfuserAgent.destroy():
    self.map_streamer.stop()
    """
    return code


if __name__ == '__main__':
    print("InterfuserMapStreamer - Real-time HD Map Streaming")
    print("=" * 60)
    print("This module integrates with InterfuserAgent to stream:")
    print("  • BEV semantic maps (current, t+1, t+2)")
    print("  • Detected objects (vehicles, pedestrians, bikes)")
    print("  • LiDAR point clouds")
    print("  • Camera feeds (front, left, right)")
    print("  • Ego vehicle state and trajectory")
    print("  • BEV features from model")
    print()
    print("Integration code:")
    print(integrate_with_agent())
