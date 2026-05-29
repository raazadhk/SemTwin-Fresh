"""
Simple BEV Map Streaming using PNG compression
"""

import socket
import numpy as np
import cv2
import time
from typing import Dict


class BEVStreamer:
    """Stream BEV maps with PNG compression"""
    
    def __init__(self, ip='255.255.255.255', port=12350):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.frame_id = 0
        print(f"BEV Streamer: {ip}:{port}")
    
    def stream(self, bev_map: np.ndarray, metadata: dict):
        """Stream BEV map as PNG"""
        try:
            # Encode as PNG (good compression)
            _, encoded = cv2.imencode('.png', bev_map)
            img_bytes = encoded.tobytes()
            
            # Add simple header: frame_id (4 bytes) + data
            import struct
            header = struct.pack('!I', self.frame_id)
            packet = header + img_bytes
            
            # Send (split if too large)
            if len(packet) < 60000:
                self.sock.sendto(packet, (self.ip, self.port))
            else:
                # Send in chunks
                chunk_size = 50000
                for i in range(0, len(packet), chunk_size):
                    chunk = packet[i:i+chunk_size]
                    self.sock.sendto(chunk, (self.ip, self.port))
                    time.sleep(0.001)
            
            self.frame_id += 1
            
        except Exception as e:
            if self.frame_id % 100 == 0:
                print(f"Stream error: {e}")
    
    def close(self):
        self.sock.close()


class BEVReceiver:
    """Receive BEV maps"""
    
    def __init__(self, port=12350):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        self.sock.settimeout(0.1)
        
        self.latest_map = None
        self.buffer = b''
        print(f"BEV Receiver: port {port}")
    
    def receive(self) -> np.ndarray:
        """Receive latest BEV map"""
        try:
            data, addr = self.sock.recvfrom(65536)
            
            # Try to decode immediately
            try:
                img = cv2.imdecode(np.frombuffer(data[4:], dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self.latest_map = img
                    return img
            except:
                pass
            
            # Otherwise buffer and try
            self.buffer += data
            if len(self.buffer) > 4:
                try:
                    img = cv2.imdecode(np.frombuffer(self.buffer[4:], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        self.latest_map = img
                        self.buffer = b''
                        return img
                except:
                    if len(self.buffer) > 200000:
                        self.buffer = b''
            
            return self.latest_map
            
        except socket.timeout:
            return self.latest_map
        except Exception as e:
            return self.latest_map
    
    def close(self):
        self.sock.close()
