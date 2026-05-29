"""
HD Map Network Broadcaster
Sends real-time HD maps from CARLA agent to remote visualizer
Add to your interfuser_agent.py
"""

import socket
import pickle
import numpy as np
import threading
from typing import Dict, Optional


class HDMapNetworkBroadcaster:
    """Broadcasts HD maps to remote visualization server"""

    def __init__(self, server_host: str, server_port: int = 9999, enabled: bool = True):
        """
        Initialize broadcaster

        Args:
            server_host: IP address of visualization server
            server_port: Port of visualization server
            enabled: Enable broadcasting
        """
        self.server_host = server_host
        self.server_port = server_port
        self.enabled = enabled
        self.socket = None
        self.connected = False
        self.send_thread = None
        self.send_queue = None

        if self.enabled:
            self._connect()

    def _connect(self):
        """Connect to remote server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            self.connected = True
            print(f"✓ Connected to HD Map Server at {self.server_host}:{self.server_port}")
        except Exception as e:
            print(f"✗ Failed to connect to HD Map Server: {e}")
            self.connected = False
            self.enabled = False

    def send_map(self, map_data: Dict, frame_id: int, timestamp: float) -> bool:
        """
        Send HD map data to visualization server

        Args:
            map_data: Map data from HDMapProcessor
                Contains: 'occupancy', 'visual', 'semantic', etc.
            frame_id: Frame number
            timestamp: Timestamp

        Returns:
            success: True if sent successfully
        """
        if not self.enabled or not self.connected:
            return False

        try:
            # Prepare message
            message = {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'occupancy': map_data.get('occupancy'),
                'visual': map_data.get('visual'),
                'fused_occ_sem': map_data.get('fused_occ_sem'),
            }

            # Serialize
            data = pickle.dumps(message)

            # Send size first
            self.socket.sendall(len(data).to_bytes(4, 'big'))

            # Send data
            self.socket.sendall(data)

            return True

        except Exception as e:
            print(f"Error sending map: {e}")
            self.connected = False
            return False

    def close(self):
        """Close connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False


def example_usage():
    """Example of how to use in interfuser_agent.py"""
    code = '''
# In interfuser_agent.py setup():
from hd_map_network_broadcaster import HDMapNetworkBroadcaster

# Initialize broadcaster (connect to remote visualizer)
self.hd_map_broadcaster = HDMapNetworkBroadcaster(
    server_host="192.168.1.100",  # IP of visualization system
    server_port=9999,
    enabled=True
)

# In interfuser_agent.py tick(), after generating maps:
if map_result and self.hd_map_broadcaster and self.hd_map_broadcaster.connected:
    self.hd_map_broadcaster.send_map(
        map_data=map_result,
        frame_id=self.step,
        timestamp=time.time()
    )

# In interfuser_agent.py destroy():
if hasattr(self, 'hd_map_broadcaster'):
    self.hd_map_broadcaster.close()
    '''
    print(code)


if __name__ == "__main__":
    example_usage()
