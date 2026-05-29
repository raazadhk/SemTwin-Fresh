"""
Simple HD Map Streaming - No Fragmentation Version
Ultra-reliable, uses only downsampling to fit in single UDP packet

This version is simpler and more reliable than fragmentation.
"""

import json
import socket
import time
import numpy as np
from typing import Dict
import os


class SimpleHDMapStreamer:
    """
    Simple UDP streaming - NO FRAGMENTATION
    Uses aggressive downsampling to keep packets small
    """
    
    def __init__(self, broadcast_ip='255.255.255.255', port=12345):
        self.broadcast_ip = broadcast_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.frame_count = 0
        
        # Always downsample to ensure small packets
        # Get from environment or default to 2
        self.downsample = int(os.environ.get('HD_MAP_DOWNSAMPLE', '2'))
        
        print(f"HD Map Streamer: Downsampling by {self.downsample}x (ensures small packets)")
        print(f"HD Map Streamer: {broadcast_ip}:{port}")
    
    def stream(self, hd_map_data: Dict, gps: np.ndarray, 
               compass: float, velocity: float):
        """
        Stream HD map with metadata
        Simple version - no fragmentation
        """
        try:
            # Prepare metadata
            metadata = {
                'frame_id': self.frame_count,
                'gps': gps.tolist() if isinstance(gps, np.ndarray) else gps,
                'heading': float(compass),
                'velocity': float(velocity),
                'timestamp': time.time(),
            }
            
            # Get maps
            static_map = hd_map_data['static_map']
            dynamic_map = hd_map_data['dynamic_map']
            
            # Downsample FIRST to reduce size
            if self.downsample > 1:
                static_map = static_map[::self.downsample, ::self.downsample]
                dynamic_map = dynamic_map[::self.downsample, ::self.downsample]
            
            # Encode as sparse (only non-zero pixels)
            sparse_data = {
                'metadata': metadata,
                'static': self._encode_sparse(static_map),
                'dynamic': self._encode_sparse(dynamic_map),
                'downsample': self.downsample
            }
            
            # Convert to JSON
            json_data = json.dumps(sparse_data)
            data_bytes = json_data.encode('utf-8')
            
            # Check size
            if len(data_bytes) > 60000:
                print(f"WARNING: Packet still too large ({len(data_bytes)} bytes). Increase HD_MAP_DOWNSAMPLE to {self.downsample + 1}")
                # Skip this frame rather than crash
                return
            
            # Send single packet
            self.sock.sendto(data_bytes, (self.broadcast_ip, self.port))
            
            self.frame_count += 1
            
        except Exception as e:
            # Suppress errors to avoid flooding console
            if self.frame_count % 100 == 0:
                print(f"Streaming error: {e}")
    
    def _encode_sparse(self, map_array: np.ndarray) -> Dict:
        """Encode map in sparse format"""
        nonzero_y, nonzero_x = np.nonzero(map_array)
        values = map_array[nonzero_y, nonzero_x]
        
        # Convert to lists (more compact than numpy arrays in JSON)
        indices = [[int(y), int(x)] for y, x in zip(nonzero_y, nonzero_x)]
        values = [int(v) for v in values]
        
        return {
            'shape': [int(map_array.shape[0]), int(map_array.shape[1])],
            'indices': indices,
            'values': values
        }
    
    def close(self):
        self.sock.close()


class SimpleHDMapReceiver:
    """
    Receive HD maps via UDP - Simple version
    No fragmentation handling
    """
    
    def __init__(self, port=12345):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        self.sock.settimeout(0.1)  # 100ms timeout
        
        self.latest_map = None
        self.error_count = 0
        self.last_good_frame = 0
        
        print(f"HD Map Receiver: port {port}")
    
    def receive(self) -> Dict:
        """Receive latest HD map (non-blocking)"""
        try:
            data, addr = self.sock.recvfrom(65536)
            
            # Try to decode
            try:
                json_str = data.decode('utf-8')
                json_data = json.loads(json_str)
                
                # Validate structure
                if 'static' not in json_data or 'dynamic' not in json_data:
                    return self.latest_map
                
                # Decode sparse maps
                static_map = self._decode_sparse(json_data['static'])
                dynamic_map = self._decode_sparse(json_data['dynamic'])
                
                # Upsample if needed
                downsample = json_data.get('downsample', 1)
                if downsample > 1:
                    import cv2
                    target_size = (static_map.shape[1] * downsample, 
                                  static_map.shape[0] * downsample)
                    static_map = cv2.resize(static_map, target_size, 
                                          interpolation=cv2.INTER_NEAREST)
                    dynamic_map = cv2.resize(dynamic_map, target_size, 
                                           interpolation=cv2.INTER_NEAREST)
                
                self.latest_map = {
                    'static_map': static_map,
                    'dynamic_map': dynamic_map,
                    'metadata': json_data.get('metadata', {})
                }
                
                self.error_count = 0
                self.last_good_frame = self.latest_map['metadata'].get('frame_id', 0)
                
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                # Corrupted packet - just skip it
                self.error_count += 1
                
                # Print status occasionally
                if self.error_count % 50 == 0:
                    print(f"Skipped {self.error_count} corrupted packets (last good frame: {self.last_good_frame})")
            
            return self.latest_map
                    
        except socket.timeout:
            return self.latest_map
        except Exception as e:
            if self.error_count % 100 == 0:
                print(f"Receive error: {e}")
            self.error_count += 1
            return self.latest_map
    
    def _decode_sparse(self, sparse_data: Dict) -> np.ndarray:
        """Decode sparse format to dense array"""
        H, W = sparse_data['shape']
        map_array = np.zeros((H, W), dtype=np.uint8)
        
        indices = sparse_data.get('indices', [])
        values = sparse_data.get('values', [])
        
        for (y, x), v in zip(indices, values):
            if 0 <= y < H and 0 <= x < W:
                map_array[y, x] = v
        
        return map_array
    
    def close(self):
        self.sock.close()


# Test
if __name__ == "__main__":
    print("Testing Simple HD Map Streaming (No Fragmentation)...")
    
    # Test encoding/decoding
    test_map = np.random.randint(0, 4, (200, 200), dtype=np.uint8)
    
    streamer = SimpleHDMapStreamer()
    
    # Downsample
    downsampled = test_map[::2, ::2]
    
    sparse = streamer._encode_sparse(downsampled)
    
    receiver = SimpleHDMapReceiver()
    decoded = receiver._decode_sparse(sparse)
    
    assert np.array_equal(downsampled, decoded), "Decode failed!"
    
    json_size = len(json.dumps(sparse).encode('utf-8'))
    
    print(f"✓ Encoding test passed")
    print(f"  Downsampled size: {downsampled.shape}")
    print(f"  JSON size: {json_size} bytes")
    
    if json_size < 60000:
        print(f"✓ Packet size OK (< 60KB)")
    else:
        print(f"⚠ Packet too large! Increase downsampling")
    
    streamer.close()
    receiver.close()
    
    print("\n✓ Simple streaming ready!")
    print("\nUsage:")
    print("  export HD_MAP_DOWNSAMPLE=2  # or 3, 4")
    print("  python your_agent.py")
