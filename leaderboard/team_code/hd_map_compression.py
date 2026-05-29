"""
HD Map Compression and Delta Encoding
Implements various compression strategies for communication-efficient transmission
"""

import numpy as np
import cv2
import zlib
from io import BytesIO
from PIL import Image
from typing import Tuple, Dict, Optional, List
from hd_map_config import MapConfig


class CompressionCodec:
    """Multi-format compression codec"""

    def __init__(self):
        self.config = MapConfig()

    def compress_jpeg(
        self,
        image: np.ndarray,
        quality: int = None
    ) -> bytes:
        """
        Compress image to JPEG format

        Args:
            image: HxWxC image
            quality: 0-100 (default: from config)

        Returns:
            compressed: JPEG bytes
        """
        if quality is None:
            quality = self.config.JPEG_QUALITY

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Ensure BGR for cv2.imwrite compatibility
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes() if success else b''

    def compress_webp(
        self,
        image: np.ndarray,
        quality: int = None
    ) -> bytes:
        """
        Compress image to WebP format (better compression than JPEG)

        Args:
            image: HxWxC image
            quality: 0-100 (default: from config)

        Returns:
            compressed: WebP bytes
        """
        if quality is None:
            quality = self.config.WEBP_QUALITY

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode('.webp', image, [cv2.IMWRITE_WEBP_QUALITY, quality])
        return buffer.tobytes() if success else b''

    def compress_png(
        self,
        image: np.ndarray,
        compression_level: int = None
    ) -> bytes:
        """
        Compress image to PNG format (lossless)

        Args:
            image: HxWxC or HxW image
            compression_level: 0-9 (default: from config)

        Returns:
            compressed: PNG bytes
        """
        if compression_level is None:
            compression_level = self.config.PNG_COMPRESSION

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode(
            '.png',
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
        )
        return buffer.tobytes() if success else b''

    def compress_zlib(
        self,
        data: bytes,
        level: int = 9
    ) -> bytes:
        """
        Compress raw data with zlib

        Args:
            data: Raw bytes
            level: Compression level 0-9 (default: 9 for max compression)

        Returns:
            compressed: Zlib compressed bytes
        """
        return zlib.compress(data, level=level)

    def decompress_zlib(self, data: bytes) -> bytes:
        """Decompress zlib data"""
        return zlib.decompress(data)

    def get_compression_ratio(
        self,
        original_size: int,
        compressed_size: int
    ) -> float:
        """Calculate compression ratio"""
        return (1 - compressed_size / original_size) * 100 if original_size > 0 else 0


class DeltaEncoder:
    """Delta encoding for incremental updates"""

    def __init__(self, threshold: float = None):
        self.config = MapConfig()
        self.threshold = threshold or self.config.DELTA_THRESHOLD
        self.previous_tiles = {}

    def should_update_tile(
        self,
        tile_id: Tuple[int, int],
        new_tile: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Determine if tile has changed enough to warrant transmission

        Args:
            tile_id: (row, col) tuple
            new_tile: New tile data

        Returns:
            (should_update, change_percentage): Boolean and change percentage
        """
        if tile_id not in self.previous_tiles:
            # First time seeing this tile
            self.previous_tiles[tile_id] = new_tile.copy()
            return True, 1.0

        previous_tile = self.previous_tiles[tile_id]

        # Compute L2 difference
        if len(new_tile.shape) == 2:
            diff = np.sqrt(np.mean((new_tile.astype(float) - previous_tile.astype(float)) ** 2))
            max_diff = 255.0
        else:
            diff = np.sqrt(np.mean((new_tile.astype(float) - previous_tile.astype(float)) ** 2))
            max_diff = 255.0

        change_percentage = diff / max_diff

        # Update previous tile
        self.previous_tiles[tile_id] = new_tile.copy()

        return change_percentage >= self.threshold, change_percentage

    def compute_delta(
        self,
        current_tile: np.ndarray,
        previous_tile: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute delta (difference) between tiles

        Args:
            current_tile: Current tile data
            previous_tile: Previous tile data

        Returns:
            (delta, mask): Delta values and significance mask
        """
        delta = current_tile.astype(int) - previous_tile.astype(int)
        mask = np.abs(delta) > 10  # Only significant changes

        return delta, mask

    def apply_delta(
        self,
        previous_tile: np.ndarray,
        delta: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply delta to previous tile to reconstruct current

        Args:
            previous_tile: Previous tile data
            delta: Delta values
            mask: Significance mask

        Returns:
            reconstructed: Reconstructed current tile
        """
        reconstructed = previous_tile.copy().astype(int)
        reconstructed[mask] += delta[mask]
        return np.clip(reconstructed, 0, 255).astype(np.uint8)

    def reset(self):
        """Reset delta encoder state"""
        self.previous_tiles = {}


class AdaptiveCompression:
    """Adaptive compression based on bandwidth constraints"""

    def __init__(self):
        self.codec = CompressionCodec()
        self.config = MapConfig()

    def choose_compression(
        self,
        image: np.ndarray,
        bandwidth_mbps: float,
        frame_rate: float = 10,  # Hz
        target_quality: str = 'high'  # 'low', 'medium', 'high'
    ) -> Dict:
        """
        Adaptively choose compression based on bandwidth

        Args:
            image: Image to compress
            bandwidth_mbps: Available bandwidth in Mbps
            frame_rate: Desired frame rate in Hz
            target_quality: Quality target ('low', 'medium', 'high')

        Returns:
            result: {
                'format': 'jpeg'|'webp'|'png',
                'quality': int,
                'compressed': bytes,
                'compression_ratio': float,
                'estimated_latency': float
            }
        """
        h, w = image.shape[:2]
        num_pixels = h * w
        num_channels = 3 if len(image.shape) == 3 else 1

        # Available bits per frame
        bits_per_second = bandwidth_mbps * 1e6
        bits_per_frame = bits_per_second / frame_rate
        bytes_per_frame = bits_per_frame / 8
        bytes_per_pixel = bytes_per_frame / (num_pixels * num_channels)

        quality_map = {
            'high': {'jpeg': 90, 'webp': 85, 'png': 9},
            'medium': {'jpeg': 75, 'webp': 70, 'png': 6},
            'low': {'jpeg': 60, 'webp': 55, 'png': 3},
        }

        qualities = quality_map[target_quality]

        # Try different formats and pick best
        results = {}

        # JPEG
        jpeg_data = self.codec.compress_jpeg(image, qualities['jpeg'])
        results['jpeg'] = {
            'data': jpeg_data,
            'size': len(jpeg_data)
        }

        # WebP
        webp_data = self.codec.compress_webp(image, qualities['webp'])
        results['webp'] = {
            'data': webp_data,
            'size': len(webp_data)
        }

        # Choose format based on size and bandwidth
        best_format = min(results.keys(), key=lambda k: results[k]['size'])
        best_data = results[best_format]['data']
        best_size = results[best_format]['size']

        original_size = num_pixels * num_channels
        compression_ratio = self.codec.get_compression_ratio(original_size, best_size)
        estimated_latency = (best_size * 8) / (bandwidth_mbps * 1e6)  # seconds

        return {
            'format': best_format,
            'quality': qualities[best_format],
            'compressed': best_data,
            'size_bytes': best_size,
            'compression_ratio': compression_ratio,
            'estimated_latency': estimated_latency,
            'original_size': original_size,
        }


class RobustTransmission:
    """Robustness under packet loss and network degradation"""

    def __init__(self):
        self.codec = CompressionCodec()

    def add_error_correction(
        self,
        data: bytes,
        redundancy_factor: float = 0.1  # 10% redundancy
    ) -> Tuple[bytes, bytes]:
        """
        Add simple redundancy for error correction

        Args:
            data: Original data
            redundancy_factor: Ratio of redundancy bytes (0.1 = 10%)

        Returns:
            (encoded_data, redundancy_data): Original + redundancy
        """
        # Simple approach: repeat critical bytes
        compressed = self.codec.compress_zlib(data)

        # Create redundancy by compressing again or adding parity bits
        redundancy_size = max(8, int(len(compressed) * redundancy_factor))
        redundancy = compressed[:redundancy_size]  # Simplified

        return compressed, redundancy

    def implement_retransmission_strategy(
        self,
        packet_loss_rate: float = 0.05  # 5% loss
    ) -> Dict:
        """
        Compute optimal retransmission strategy for given packet loss

        Args:
            packet_loss_rate: Estimated packet loss rate (0-1)

        Returns:
            strategy: {
                'num_redundant_packets': int,
                'timeout_ms': float,
                'max_retries': int
            }
        """
        # Expected number of transmissions needed
        if packet_loss_rate >= 1.0:
            return {'num_redundant_packets': 0, 'timeout_ms': 1000, 'max_retries': 0}

        expected_transmissions = 1 / (1 - packet_loss_rate)
        redundant_packets = max(1, int(expected_transmissions - 1))

        timeout = 100 * expected_transmissions  # Scale timeout with loss
        max_retries = int(np.log(1e-6) / np.log(packet_loss_rate))  # Until 1 in million

        return {
            'num_redundant_packets': int(redundant_packets),
            'timeout_ms': float(timeout),
            'max_retries': int(max_retries),
        }


if __name__ == "__main__":
    print("HD Map Compression and Delta Encoding")
    print("Available classes:")
    print("  - CompressionCodec: JPEG, WebP, PNG, Zlib compression")
    print("  - DeltaEncoder: Incremental tile updates")
    print("  - AdaptiveCompression: Bandwidth-aware compression")
    print("  - RobustTransmission: Error correction and retransmission")
