"""
Efficient BEV Semantic Map Streaming
Compress and stream semantic segmentation maps over network
"""

import json
import socket
import time
import numpy as np
from typing import Dict, Tuple
import zlib
import struct


class BEVSemanticStreamer:
    """
    Stream BEV semantic maps efficiently
    Uses RLE compression for semantic maps (very effective!)
    """
    
    def __init__(self, broadcast_ip='255.255.255.255', port=12346):
        self.broadcast_ip = broadcast_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.frame_count = 0
        
        print(f"BEV Semantic Streamer: {broadcast_ip}:{port}")
    
    def stream(self, semantic_data: Dict, gps: np.ndarray, 
               compass: float, velocity: float):
        """
        Stream semantic map
        
        Args:
            semantic_data: Output from BEVSemanticMapper.generate_bev_semantic_map()
            gps, compass, velocity: Vehicle state
        """
        try:
            semantic_map = semantic_data['semantic_map']
            
            # Compress using RLE (Run-Length Encoding)
            compressed = self._compress_rle(semantic_map)
            
            # Metadata
            metadata = {
                'frame_id': self.frame_count,
                'gps': gps.tolist() if isinstance(gps, np.ndarray) else gps,
                'heading': float(compass),
                'velocity': float(velocity),
                'timestamp': time.time(),
                'shape': list(semantic_map.shape)
            }
            
            # Package
            data = {
                'metadata': metadata,
                'semantic': compressed
            }
            
            # Send
            json_data = json.dumps(data)
            data_bytes = json_data.encode('utf-8')
            
            if len(data_bytes) > 60000:
                print(f"Warning: Packet size {len(data_bytes)} bytes (RLE should make this much smaller!)")
            
            self.sock.sendto(data_bytes, (self.broadcast_ip, self.port))
            
            self.frame_count += 1
            
        except Exception as e:
            if self.frame_count % 100 == 0:
                print(f"Streaming error: {e}")
    
    def _compress_rle(self, semantic_map: np.ndarray) -> Dict:
        """
        Run-Length Encoding compression
        Perfect for semantic maps with large uniform regions
        """
        flat = semantic_map.flatten()
        
        rle_values = []
        rle_lengths = []
        
        current_val = int(flat[0])
        current_len = 1
        
        for val in flat[1:]:
            if val == current_val:
                current_len += 1
            else:
                rle_values.append(current_val)
                rle_lengths.append(current_len)
                current_val = int(val)
                current_len = 1
        
        # Add last run
        rle_values.append(current_val)
        rle_lengths.append(current_len)
        
        return {
            'values': rle_values,
            'lengths': rle_lengths
        }
    
    def close(self):
        self.sock.close()


class BEVSemanticReceiver:
    """
    Receive BEV semantic maps
    """
    
    def __init__(self, port=12346):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        self.sock.settimeout(0.1)
        
        self.latest_map = None
        self.error_count = 0
        
        print(f"BEV Semantic Receiver: port {port}")
    
    def receive(self) -> Dict:
        """Receive latest semantic map"""
        try:
            data, addr = self.sock.recvfrom(65536)
            
            try:
                json_str = data.decode('utf-8')
                json_data = json.loads(json_str)
                
                # Validate
                if 'semantic' not in json_data or 'metadata' not in json_data:
                    return self.latest_map
                
                # Decompress
                semantic_map = self._decompress_rle(
                    json_data['semantic'],
                    tuple(json_data['metadata']['shape'])
                )
                
                self.latest_map = {
                    'semantic_map': semantic_map,
                    'metadata': json_data['metadata']
                }
                
                self.error_count = 0
                
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.error_count += 1
                if self.error_count % 50 == 0:
                    print(f"Skipped {self.error_count} corrupted packets")
            
            return self.latest_map
            
        except socket.timeout:
            return self.latest_map
        except Exception as e:
            if self.error_count % 100 == 0:
                print(f"Receive error: {e}")
            self.error_count += 1
            return self.latest_map
    
    def _decompress_rle(self, rle_data: Dict, shape: Tuple) -> np.ndarray:
        """Decompress RLE to semantic map"""
        values = rle_data['values']
        lengths = rle_data['lengths']
        
        # Reconstruct flat array
        flat = []
        for val, length in zip(values, lengths):
            flat.extend([val] * length)
        
        # Reshape
        semantic_map = np.array(flat, dtype=np.uint8).reshape(shape)
        
        return semantic_map
    
    def close(self):
        self.sock.close()


# Test
if __name__ == "__main__":
    print("Testing BEV Semantic Streaming...")
    
    # Create sample semantic map
    semantic_map = np.zeros((400, 400), dtype=np.uint8)
    semantic_map[100:300, 150:250] = 0  # road
    semantic_map[150:160, :] = 1  # lane
    semantic_map[200:220, 180:200] = 2  # vehicle
    semantic_map[250:260, 200:210] = 3  # pedestrian
    
    # Test compression
    streamer = BEVSemanticStreamer()
    compressed = streamer._compress_rle(semantic_map)
    
    receiver = BEVSemanticReceiver()
    decompressed = receiver._decompress_rle(compressed, semantic_map.shape)
    
    assert np.array_equal(semantic_map, decompressed), "RLE failed!"
    
    # Check compression ratio
    raw_size = semantic_map.nbytes
    compressed_size = len(json.dumps(compressed).encode('utf-8'))
    
    print(f"✓ RLE compression test passed")
    print(f"  Raw size: {raw_size} bytes")
    print(f"  Compressed: {compressed_size} bytes")
    print(f"  Ratio: {raw_size/compressed_size:.1f}x")
    
    streamer.close()
    receiver.close()
    
    print("\n✓ BEV Semantic Streaming ready!")
