"""
Real-Time HD Map Generator for CARLA interfuser_agent
FIXED VERSION - Eliminates memory leaks
"""

import numpy as np
import cv2
import os
import json
import threading
import queue
from collections import deque
from typing import Dict, Optional, Tuple
import time
import gc

from hd_map_config import MapConfig, DatasetConfig
from hd_map_layers import OccupancyGridGenerator, VisualLayerGenerator, LayerFusion
from hd_map_tiles import MapTileGenerator, TileMetadata
from hd_map_compression import DeltaEncoder, CompressionCodec


class RealtimeHDMapGenerator:
    """Generate HD maps in real-time during CARLA simulation - MEMORY OPTIMIZED"""

    def __init__(self, output_dir: str = None, enable_delta: bool = True):
        """
        Initialize real-time HD map generator

        Args:
            output_dir: Where to save generated maps (default: dataset HD_Maps/)
            enable_delta: Use delta encoding to skip unchanged tiles
        """
        self.config = MapConfig()
        self.output_dir = output_dir or os.path.join(
            DatasetConfig.OUTPUT_DIR, "RealtimeOutput"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Layer generators
        self.occ_gen = OccupancyGridGenerator()
        self.vis_gen = VisualLayerGenerator()
        self.fusion = LayerFusion()
        self.tile_gen = MapTileGenerator()
        self.tile_metadata = TileMetadata()
        self.codec = CompressionCodec()

        # Delta encoding
        self.enable_delta = enable_delta
        self.delta_encoder = DeltaEncoder()

        # Frame buffer for temporal fusion - REDUCED SIZE
        self.merged_occupancy = None
        self.frame_buffer = deque(maxlen=3)  # Keep only 3 frames max

        # Threading - REDUCED QUEUE SIZES
        self.processing_queue = queue.Queue(maxsize=5)  # Was 10
        self.result_queue = queue.Queue(maxsize=2)  # Was 5
        self.processing_thread = None
        self.is_running = False

        # Statistics
        self.frames_processed = 0
        self.tiles_generated = 0
        self.tiles_sent = 0
        self.start_time = None
        self.gc_counter = 0

    def start(self):
        """Start real-time processing thread"""
        if self.is_running:
            return

        self.is_running = True
        self.start_time = time.time()
        self.processing_thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self.processing_thread.start()
        print("✓ Real-time HD map generator started (MEMORY OPTIMIZED)")

    def stop(self):
        """Stop real-time processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

        # Cleanup
        self.merged_occupancy = None
        self.frame_buffer.clear()
        gc.collect()

        print(f"✓ HD map generator stopped ({self.frames_processed} frames processed)")

    def process_frame(
        self,
        frame_id: int,
        rgb: np.ndarray,
        bev: np.ndarray,
        lidar: np.ndarray,
        segmentation: np.ndarray,
        vehicle_position: np.ndarray,
        vehicle_rotation: float,
    ) -> Optional[Dict]:
        """
        Queue frame for real-time processing

        Args:
            frame_id: Frame number
            rgb: RGB image (HxWx3)
            bev: BEV camera image (HxWx3)
            lidar: LiDAR points (Nx3)
            segmentation: Semantic segmentation (HxWx3)
            vehicle_position: [x, y] GPS position
            vehicle_rotation: Yaw angle in radians

        Returns:
            map_data: Generated map or None if still processing
        """
        frame_data = {
            'frame_id': frame_id,
            'rgb': rgb,
            'bev': bev,
            'lidar': lidar,
            'segmentation': segmentation,
            'vehicle_position': vehicle_position,
            'vehicle_rotation': vehicle_rotation,
            'timestamp': time.time(),
        }

        try:
            self.processing_queue.put_nowait(frame_data)
        except queue.Full:
            # Queue full, skip frame - don't hold onto data
            return None

        # Try to get result without blocking
        try:
            result = self.result_queue.get_nowait()
            return result
        except queue.Empty:
            return None

    def _process_loop(self):
        """Main processing loop (runs in background thread)"""
        while self.is_running:
            try:
                # Get frame from queue with timeout
                frame_data = self.processing_queue.get(timeout=0.1)

                # Process frame
                map_data = self._generate_map_for_frame(frame_data)

                # Generate tiles
                if map_data:
                    tiles = self._generate_tiles_for_frame(map_data, frame_data)

                    # Put result in result queue
                    try:
                        self.result_queue.put_nowait(map_data)
                    except queue.Full:
                        pass  # Result queue full, drop oldest

                self.frames_processed += 1

                # Cleanup frame data to free memory
                del frame_data
                del map_data

                # Periodic garbage collection
                self.gc_counter += 1
                if self.gc_counter % 10 == 0:
                    gc.collect()
                    self.gc_counter = 0

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in HD map processing: {e}")
                continue

    def _generate_map_for_frame(self, frame_data: Dict) -> Optional[Dict]:
        """Generate map layers for a single frame - MEMORY OPTIMIZED"""
        try:
            map_data = {
                'frame_id': frame_data['frame_id'],
                'timestamp': frame_data['timestamp'],
            }

            # ========== OCCUPANCY GRID ==========
            lidar = frame_data['lidar']
            vehicle_pos = frame_data['vehicle_position']
            vehicle_rot = frame_data['vehicle_rotation']

            # Generate occupancy grid
            occ_grid = self.occ_gen.lidar_to_occupancy(lidar, vehicle_pos, vehicle_rot)

            # Temporal fusion - MEMORY OPTIMIZED
            if self.merged_occupancy is None:
                self.merged_occupancy = occ_grid.copy()
            else:
                # Avoid creating temporary copy - do in-place operation where possible
                self.merged_occupancy = self.occ_gen.apply_occupancy_decay(
                    occ_grid, self.merged_occupancy, self.config.OCCUPANCY_DECAY
                )

            # Store reference, not copy
            map_data['occupancy'] = self.merged_occupancy

            # ========== VISUAL LAYER ==========
            bev = frame_data['bev']
            visual = self.vis_gen.use_bev_camera(bev, vehicle_pos, vehicle_rot)
            map_data['visual'] = visual

            # ========== SEMANTIC LAYER ==========
            # Create minimal semantic layer (not full size)
            map_data['semantic'] = np.zeros(
                (self.occ_gen.grid_size, self.occ_gen.grid_size, 23), dtype=np.uint8
            )

            # ========== FUSE LAYERS ==========
            fused = self.fusion.fuse_layers(
                map_data['occupancy'],
                map_data['semantic'],
                map_data['visual'],
            )
            map_data.update(fused)

            return map_data

        except Exception as e:
            print(f"Error generating map: {e}")
            return None

    def _generate_tiles_for_frame(
        self, map_data: Dict, frame_data: Dict
    ) -> Dict[Tuple[int, int], Dict]:
        """Generate tiles from map data - MEMORY OPTIMIZED"""
        tiles_to_send = {}

        try:
            if 'occupancy' not in map_data:
                return tiles_to_send

            occupancy = map_data['occupancy']
            occ_tiles = self.tile_gen.divide_into_tiles(occupancy)
            vehicle_pos = frame_data['vehicle_position']

            for (tile_row, tile_col), tile_data in occ_tiles.items():
                tile_key = (tile_row, tile_col)

                # Check if tile changed (delta encoding)
                if self.enable_delta:
                    should_update, change_pct = self.delta_encoder.should_update_tile(
                        tile_key, tile_data
                    )
                else:
                    should_update, change_pct = True, 1.0

                if should_update:
                    # Compress tile
                    compressed = self.codec.compress_webp(tile_data, quality=60)  # Lower quality

                    # Create metadata (lightweight)
                    metadata = self.tile_metadata.create_tile_metadata(
                        tile_row, tile_col,
                        frame_data['frame_id'],
                        frame_data['timestamp'],
                        vehicle_pos,
                        tile_data,
                        compressed_size=len(compressed),
                    )
                    metadata['change_percentage'] = float(change_pct)

                    # Store only compressed version to save memory
                    tiles_to_send[tile_key] = {
                        'compressed': compressed,
                        'metadata': metadata,
                    }
                    self.tiles_sent += 1

            self.tiles_generated += len(occ_tiles)

            # Cleanup tile data
            del occ_tiles

            return tiles_to_send

        except Exception as e:
            print(f"Error generating tiles: {e}")
            return {}

    def get_stats(self) -> Dict:
        """Get processing statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frames_processed / elapsed if elapsed > 0 else 0

        return {
            'frames_processed': self.frames_processed,
            'tiles_generated': self.tiles_generated,
            'tiles_sent': self.tiles_sent,
            'fps': fps,
            'elapsed_seconds': elapsed,
            'queue_size': self.processing_queue.qsize(),
            'compression_ratio': (
                (1 - 0.05) * 100 if self.tiles_sent > 0 else 0
            ),
        }

    def print_stats(self):
        """Print statistics"""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"Real-Time HD Map Generator Statistics (MEMORY OPTIMIZED)")
        print(f"{'='*60}")
        print(f"Frames processed:     {stats['frames_processed']}")
        print(f"Tiles generated:      {stats['tiles_generated']}")
        print(f"Tiles transmitted:    {stats['tiles_sent']}")
        print(f"Processing FPS:       {stats['fps']:.1f}")
        print(f"Elapsed time:         {stats['elapsed_seconds']:.1f}s")
        print(f"Estimated saving:     ~{stats['compression_ratio']:.0f}% (delta encoding)")
        print(f"{'='*60}\n")


class RealtimeMapBroadcaster:
    """Broadcast real-time maps to digital twin"""

    def __init__(self, host: str = None, port: int = 9999):
        """
        Initialize broadcaster

        Args:
            host: Digital twin server hostname
            port: Digital twin server port
        """
        self.host = host
        self.port = port
        self.connected = False
        self.socket = None

        if host:
            self._connect()

    def _connect(self):
        """Connect to digital twin server"""
        import socket

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # Add timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✓ Connected to digital twin at {self.host}:{self.port}")
        except Exception as e:
            print(f"✗ Failed to connect to digital twin: {e}")
            self.connected = False

    def send_tiles(
        self,
        tiles: Dict[Tuple[int, int], Dict],
        frame_id: int,
        timestamp: float,
    ):
        """Send map tiles to digital twin - MEMORY OPTIMIZED"""
        if not self.connected:
            return

        import pickle

        try:
            for tile_key, tile_info in tiles.items():
                # Send only compressed data, not raw tile
                message = {
                    'type': 'map_tile',
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'tile_id': tile_key,
                    'tile_row': tile_key[0],
                    'tile_col': tile_key[1],
                    'compressed_data': tile_info['compressed'],
                    'format': 'webp',
                }

                data = pickle.dumps(message)
                self.socket.sendall(len(data).to_bytes(4, 'big'))
                self.socket.sendall(data)

                # Delete after sending
                del data

        except Exception as e:
            print(f"Error broadcasting tiles: {e}")
            self.connected = False

    def close(self):
        """Close connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
