import os
import json
import datetime
import pathlib
import time
import imp
import cv2
import carla
from collections import deque
import socket
import struct
import pickle
import random
import threading

import torch
import numpy as np
from PIL import Image
from easydict import EasyDict
from io import BytesIO

from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
from timm.models import create_model
from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.interfuser_controller import InterfuserController
from team_code.render import render, render_self_car, render_waypoints
from team_code.tracker import Tracker

import math
import yaml

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------
# Digital Twin Broadcaster
# ----------------------------------------------------------------------
class DigitalTwinBroadcaster:
    """Broadcasts vehicle state to digital twin computer"""
    def __init__(self, twin_host=None, twin_port=9999, enabled=True):
        self.enabled = enabled and twin_host is not None
        self.socket = None
        self.connected = False

        if self.enabled:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5.0)
                self.socket.connect((twin_host, twin_port))
                self.connected = True
                print(f"✓ Connected to Digital Twin at {twin_host}:{twin_port}")
            except Exception as e:
                print(f"✗ Failed to connect to Digital Twin: {e}")
                self.enabled = False

    def send_state(self, state_data):
        """Send current state to digital twin"""
        if not self.enabled or not self.connected:
            return

        try:
            data = pickle.dumps(state_data)
            self.socket.sendall(len(data).to_bytes(4, 'big'))
            self.socket.sendall(data)
        except Exception as e:
            print(f"Error sending to digital twin: {e}")
            self.connected = False

    def close(self):
        if self.socket:
            try:
                self.socket.close()
            except:
                pass


def compress_image(image, quality=60):
    """Compress image to JPEG bytes for efficient transmission"""
    if image is None:
        return None

    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_img = Image.fromarray(image)
    else:
        pil_img = image

    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    return buffer.getvalue()


# ----------------------------------------------------------------------
# Dataset configuration
# ----------------------------------------------------------------------
def check_disk_space(path):
    """Check available disk space"""
    try:
        stat = os.statvfs(path)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        used_percent = ((total_gb - free_gb) / total_gb) * 100
        return free_gb, total_gb, used_percent
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
        return 0, 0, 0


DATASET_DIR = os.environ.get("DATASET_DIR", "/new_ssd/interfuser_complete_dataset")

if not os.path.exists(DATASET_DIR):
    print(f"Creating dataset directory: {DATASET_DIR}")
    os.makedirs(DATASET_DIR, exist_ok=True)

free, total, used_pct = check_disk_space(DATASET_DIR)
print(f"\n{'='*60}")
print(f"DATASET STORAGE CONFIGURATION")
print(f"{'='*60}")
print(f"Location: {DATASET_DIR}")
print(f"Available: {free:.2f} GB / {total:.2f} GB ({used_pct:.1f}% used)")
print(f"Estimated capacity: ~{int(free * 1000 / 0.5)} frames (@ 0.5 MB/frame)")
if free < 10:
    print("⚠️  WARNING: Less than 10 GB free space!")
elif free > 100:
    print("✓ Sufficient storage available")
print(f"{'='*60}\n")

FRAME_DIRS = {
    "rgb": os.path.join(DATASET_DIR, "RGB"),
    "rgb_left": os.path.join(DATASET_DIR, "RGB_Left"),
    "rgb_right": os.path.join(DATASET_DIR, "RGB_Right"),
    "lidar": os.path.join(DATASET_DIR, "LiDAR"),
    "measurements": os.path.join(DATASET_DIR, "Measurements"),
    "bev_features": os.path.join(DATASET_DIR, "BEV_Features"),
    "metadata": os.path.join(DATASET_DIR, "Metadata"),

    # Semantic Segmentation (front-view)
    "seg_front": os.path.join(DATASET_DIR, "Segmentation_Front"),
    "seg_left": os.path.join(DATASET_DIR, "Segmentation_Left"),
    "seg_right": os.path.join(DATASET_DIR, "Segmentation_Right"),
    "seg_colored_front": os.path.join(DATASET_DIR, "Segmentation_Colored_Front"),

    # Pseudo Instance Segmentation (front-view)
    "inst_front": os.path.join(DATASET_DIR, "Instance_Front"),
    "inst_left": os.path.join(DATASET_DIR, "Instance_Left"),
    "inst_right": os.path.join(DATASET_DIR, "Instance_Right"),
    "inst_colored_front": os.path.join(DATASET_DIR, "Instance_Colored_Front"),
    "inst_analysis": os.path.join(DATASET_DIR, "Instance_Analysis"),

    # Tracking (front-view)
    "tracking": os.path.join(DATASET_DIR, "Tracking"),
    "tracking_visualization": os.path.join(DATASET_DIR, "Tracking_Visualization"),
    "trajectories": os.path.join(DATASET_DIR, "Trajectories"),
}

for dir_path in FRAME_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

print(f"✓ All dataset directories created under: {DATASET_DIR}\n")


# ----------------------------------------------------------------------
# Helper Functions: semantic + instance + tracking
# ----------------------------------------------------------------------
def convert_segmentation_to_rgb(seg_image):
    """Convert CARLA semantic segmentation to colored RGB image"""
    color_map = {
        0: [0, 0, 0], 1: [70, 70, 70], 2: [100, 40, 40], 3: [55, 90, 80],
        4: [220, 20, 60], 5: [153, 153, 153], 6: [157, 234, 50],
        7: [128, 64, 128], 8: [244, 35, 232], 9: [107, 142, 35],
        10: [0, 0, 142], 11: [102, 102, 156], 12: [220, 220, 0],
        13: [70, 130, 180], 14: [81, 0, 81], 15: [150, 100, 100],
        16: [230, 150, 140], 17: [180, 165, 180], 18: [250, 170, 30],
        19: [110, 190, 160], 20: [170, 120, 50], 21: [45, 60, 150],
        22: [145, 170, 100],
    }

    seg_class_ids = seg_image[:, :, 2]
    h, w = seg_class_ids.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        mask = seg_class_ids == class_id
        rgb_image[mask] = color

    return rgb_image


def create_pseudo_instances_from_semantic(semantic_image):
    semantic_ids = semantic_image[:, :, 2]
    h, w = semantic_ids.shape
    instance_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Important classes for instance-level boxes
    important_classes = [4, 10, 12, 18]  # pedestrians, vehicles, signs, lights
    instance_id = 1
    instance_map = {}

    for class_id in important_classes:
        class_mask = (semantic_ids == class_id).astype(np.uint8)
        if np.sum(class_mask) == 0:
            continue

        num_labels, labels = cv2.connectedComponents(class_mask)

        for label in range(1, num_labels):
            component_mask = (labels == label)
            if np.sum(component_mask) < 100:
                continue

            instance_image[component_mask, 2] = instance_id % 256
            instance_image[component_mask, 1] = instance_id // 256
            instance_image[component_mask, 0] = 0

            instance_map[instance_id] = class_id
            instance_id += 1

    return instance_image, instance_map


def convert_instance_to_rgb(instance_image):
    instance_ids = instance_image[:, :, 2].astype(np.uint32) + \
                   instance_image[:, :, 1].astype(np.uint32) * 256

    h, w = instance_ids.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(instance_ids)

    for instance_id in unique_ids:
        if instance_id == 0:
            continue
        np.random.seed(int(instance_id))
        color = [np.random.randint(50, 255),
                 np.random.randint(50, 255),
                 np.random.randint(50, 255)]
        mask = instance_ids == instance_id
        rgb_image[mask] = color

    return rgb_image


def analyze_instances(instance_image, semantic_image):
    instance_ids = instance_image[:, :, 2].astype(np.uint32) + \
                   instance_image[:, :, 1].astype(np.uint32) * 256

    semantic_ids = semantic_image[:, :, 2]
    unique_instances = np.unique(instance_ids)

    object_counts = {
        'vehicles': 0, 'pedestrians': 0, 'traffic_lights': 0,
        'traffic_signs': 0, 'total_objects': 0
    }

    instances_info = []

    for instance_id in unique_instances:
        if instance_id == 0:
            continue

        mask = instance_ids == instance_id
        semantic_classes = semantic_ids[mask]
        if len(semantic_classes) == 0:
            continue

        unique_classes, counts = np.unique(semantic_classes, return_counts=True)
        class_id = unique_classes[np.argmax(counts)]
        pixel_count = np.sum(mask)

        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue

        bbox = {
            'x_min': int(np.min(cols)), 'y_min': int(np.min(rows)),
            'x_max': int(np.max(cols)), 'y_max': int(np.max(rows))
        }

        class_name = 'unknown'
        if class_id == 10:
            object_counts['vehicles'] += 1
            class_name = 'vehicle'
        elif class_id == 4:
            object_counts['pedestrians'] += 1
            class_name = 'pedestrian'
        elif class_id == 18:
            object_counts['traffic_lights'] += 1
            class_name = 'traffic_light'
        elif class_id == 12:
            object_counts['traffic_signs'] += 1
            class_name = 'traffic_sign'

        object_counts['total_objects'] += 1

        instances_info.append({
            'instance_id': int(instance_id),
            'class_id': int(class_id),
            'class_name': class_name,
            'pixel_count': int(pixel_count),
            'bbox': bbox,
            'center_x': int((bbox['x_min'] + bbox['x_max']) / 2),
            'center_y': int((bbox['y_min'] + bbox['y_max']) / 2)
        })

    return {'counts': object_counts, 'instances': instances_info}


def draw_instance_boxes(rgb_image, instances_info):
    image_with_boxes = rgb_image.copy()

    class_colors = {
        'vehicle': (0, 0, 255),
        'pedestrian': (255, 0, 0),
        'traffic_light': (0, 255, 255),
        'traffic_sign': (0, 255, 0),
        'unknown': (128, 128, 128)
    }

    for instance in instances_info['instances']:
        bbox = instance['bbox']
        class_name = instance['class_name']
        color = class_colors.get(class_name, (255, 255, 255))

        cv2.rectangle(image_with_boxes, (bbox['x_min'], bbox['y_min']),
                      (bbox['x_max'], bbox['y_max']), color, 2)

        label = f"{class_name}_{instance['instance_id']}"
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        cv2.rectangle(
            image_with_boxes,
            (bbox['x_min'], bbox['y_min'] - label_size[1] - 5),
            (bbox['x_min'] + label_size[0], bbox['y_min']),
            color, -1
        )

        cv2.putText(
            image_with_boxes, label,
            (bbox['x_min'], bbox['y_min'] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1
        )

    return image_with_boxes


def draw_tracked_objects(rgb_image, tracking_results, show_trails=True):
    image_with_tracks = rgb_image.copy()

    class_colors = {
        'vehicle': (0, 0, 255), 'pedestrian': (255, 0, 0),
        'traffic_light': (0, 255, 255), 'traffic_sign': (0, 255, 0),
        'unknown': (128, 128, 128)
    }

    for track_id, track_info in tracking_results.items():
        bbox = track_info['bbox']
        class_name = track_info['class_name']
        center = track_info['center']
        velocity = track_info['velocity']
        frames_tracked = track_info['frames_tracked']

        color = class_colors.get(class_name, (255, 255, 255))

        cv2.rectangle(
            image_with_tracks,
            (bbox['x_min'], bbox['y_min']),
            (bbox['x_max'], bbox['y_max']),
            color, 3
        )

        cv2.circle(image_with_tracks, tuple(center), 5, color, -1)

        if velocity['speed'] > 1:
            arrow_end = (int(center[0] + velocity['vx'] * 10),
                         int(center[1] + velocity['vy'] * 10))
            cv2.arrowedLine(
                image_with_tracks, tuple(center), arrow_end,
                (0, 255, 0), 2, tipLength=0.3
            )

        label = f"ID:{track_id} {class_name}"
        info = f"F:{frames_tracked} S:{velocity['speed']:.1f}"

        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        info_size, _ = cv2.getTextSize(
            info, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )

        cv2.rectangle(
            image_with_tracks,
            (bbox['x_min'],
             bbox['y_min'] - label_size[1] - info_size[1] - 10),
            (bbox['x_min'] +
             max(label_size[0], info_size[0]) + 5,
             bbox['y_min']),
            color, -1
        )

        cv2.putText(
            image_with_tracks, label,
            (bbox['x_min'] + 2, bbox['y_min'] - info_size[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2
        )

        cv2.putText(
            image_with_tracks, info,
            (bbox['x_min'] + 2, bbox['y_min'] - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (255, 255, 255), 1
        )

    return image_with_tracks


class InstanceTracker:
    """Track objects across frames"""
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.frame_count = 0

    def register(self, instance_id, class_name, center, bbox, additional_info=None):
        self.objects[self.next_object_id] = {
            'instance_id': instance_id,
            'class_name': class_name,
            'first_seen_frame': self.frame_count,
            'last_seen_frame': self.frame_count,
            'total_frames_visible': 1,
            'history': [{
                'frame': self.frame_count,
                'center': center,
                'bbox': bbox,
                'additional_info': additional_info or {}
            }]
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        self.frame_count += 1

        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection['instance_id'], detection['class_name'],
                              detection['center'], detection['bbox'],
                              detection.get('additional_info'))

            tracking_results = {}
            for object_id, obj_info in self.objects.items():
                tracking_results[object_id] = {
                    'instance_id': obj_info['instance_id'],
                    'class_name': obj_info['class_name'],
                    'center': obj_info['history'][-1]['center'],
                    'bbox': obj_info['history'][-1]['bbox'],
                    'track_id': object_id,
                    'frames_tracked': obj_info['total_frames_visible'],
                    'first_seen': obj_info['first_seen_frame'],
                    'velocity': {'vx': 0, 'vy': 0, 'speed': 0}
                }
            return tracking_results

        object_ids = list(self.objects.keys())
        previous_centers = np.array(
            [self.objects[oid]['history'][-1]['center'] for oid in object_ids]
        )
        current_centers = np.array([d['center'] for d in detections])

        distances = np.linalg.norm(
            previous_centers[:, np.newaxis] - current_centers[np.newaxis, :], axis=2
        )

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        matches = []

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue
            matches.append((row, col))
            used_rows.add(row)
            used_cols.add(col)

        for (row, col) in matches:
            object_id = object_ids[row]
            detection = detections[col]

            self.objects[object_id]['last_seen_frame'] = self.frame_count
            self.objects[object_id]['total_frames_visible'] += 1
            self.objects[object_id]['history'].append({
                'frame': self.frame_count,
                'center': detection['center'],
                'bbox': detection['bbox'],
                'additional_info': detection.get('additional_info', {})
            })
            self.disappeared[object_id] = 0

        unmatched_rows = set(range(len(object_ids))) - used_rows
        for row in unmatched_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unmatched_cols = set(range(len(detections))) - used_cols
        for col in unmatched_cols:
            detection = detections[col]
            self.register(detection['instance_id'], detection['class_name'],
                          detection['center'], detection['bbox'],
                          detection.get('additional_info'))

        tracking_results = {}
        for object_id, obj_info in self.objects.items():
            if self.disappeared[object_id] == 0:
                tracking_results[object_id] = {
                    'instance_id': obj_info['instance_id'],
                    'class_name': obj_info['class_name'],
                    'center': obj_info['history'][-1]['center'],
                    'bbox': obj_info['history'][-1]['bbox'],
                    'track_id': object_id,
                    'frames_tracked': obj_info['total_frames_visible'],
                    'first_seen': obj_info['first_seen_frame'],
                    'velocity': self._estimate_velocity(obj_info)
                }

        return tracking_results

    def _estimate_velocity(self, obj_info, window=5):
        history = obj_info['history']
        if len(history) < 2:
            return {'vx': 0, 'vy': 0, 'speed': 0}

        recent = history[-min(window, len(history)):]
        if len(recent) < 2:
            return {'vx': 0, 'vy': 0, 'speed': 0}

        start_center = recent[0]['center']
        end_center = recent[-1]['center']
        dt = recent[-1]['frame'] - recent[0]['frame']
        if dt == 0:
            return {'vx': 0, 'vy': 0, 'speed': 0}

        vx = (end_center[0] - start_center[0]) / dt
        vy = (end_center[1] - start_center[1]) / dt
        speed = np.sqrt(vx**2 + vy**2)

        return {'vx': float(vx), 'vy': float(vy), 'speed': float(speed)}

    def export_tracks_to_json(self, filepath):
        export_data = {
            'total_frames': self.frame_count,
            'total_objects_tracked': len(self.objects),
            'tracks': {}
        }

        for obj_id, obj_info in self.objects.items():
            export_data['tracks'][str(obj_id)] = {
                'instance_id': obj_info['instance_id'],
                'class_name': obj_info['class_name'],
                'first_seen_frame': obj_info['first_seen_frame'],
                'last_seen_frame': obj_info['last_seen_frame'],
                'total_frames_visible': obj_info['total_frames_visible'],
                'history': obj_info['history']
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


def save_sensor_data_to_disk(step, tick_data, input_data, frame_dirs):
    frame_id = f"{step:06d}"

    if 'raw_lidar' in tick_data:
        lidar_path = os.path.join(frame_dirs['lidar'], f"{frame_id}.npy")
        np.save(lidar_path, tick_data['raw_lidar'])

    measurements = {
        'gps': tick_data['gps'].tolist() if hasattr(tick_data['gps'], 'tolist') else list(tick_data['gps']),
        'compass': float(tick_data['compass']),
        'speed': float(tick_data['speed']),
        'measurements': tick_data['measurements'],
        'target_point': tick_data['target_point'].tolist() if hasattr(tick_data['target_point'], 'tolist') else list(tick_data['target_point']),
        'next_command': int(tick_data['next_command']),
        'frame_id': frame_id,
        'timestamp': input_data['rgb'][0] if 'rgb' in input_data else 0.0,
        'has_segmentation': True,
        'segmentation_channels': ['front', 'left', 'right'],
        'has_instance_segmentation': True,
        'instance_type': 'pseudo',
        'instance_counts': tick_data.get('instance_analysis', {}).get('counts', {}),
    }

    measurements_path = os.path.join(frame_dirs['measurements'], f"{frame_id}.json")
    with open(measurements_path, 'w') as f:
        json.dump(measurements, f, indent=2)


# ----------------------------------------------------------------------
# Display Interface
# ----------------------------------------------------------------------
class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        rgb = input_data['rgb']
        rgb_left = input_data['rgb_left']
        rgb_right = input_data['rgb_right']
        rgb_focus = input_data['rgb_focus']
        map_surround = input_data['map']

        surface = np.zeros((600, 1200, 3), np.uint8)
        surface[:, :800] = rgb
        surface[:400, 800:1200] = map_surround
        surface[400:600, 800:1000] = input_data['map_t1']
        surface[400:600, 1000:1200] = input_data['map_t2']
        surface[:150, :200] = rgb_left
        surface[:150, 600:800] = rgb_right
        surface[:150, 325:475] = rgb_focus

        surface = cv2.putText(surface, input_data['control'], (20, 580),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][0], (20, 560),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][1], (20, 540),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        surface = cv2.putText(surface, input_data['time'], (20, 520),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        surface = cv2.putText(surface, 'Left  View', (40, 135),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        surface = cv2.putText(surface, 'Focus View', (335, 135),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        surface = cv2.putText(surface, 'Right View', (640, 135),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        surface = cv2.putText(surface, 'Future Prediction', (940, 420),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        surface = cv2.putText(surface, 't', (1160, 385),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        surface = cv2.putText(surface, '0', (1170, 385),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        surface = cv2.putText(surface, 't', (960, 585),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        surface = cv2.putText(surface, '1', (970, 585),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        surface = cv2.putText(surface, 't', (1160, 585),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        surface = cv2.putText(surface, '2', (1170, 585),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        surface[:150, 198:202] = 0
        surface[:150, 323:327] = 0
        surface[:150, 473:477] = 0
        surface[:150, 598:602] = 0
        surface[148:152, :200] = 0
        surface[148:152, 325:475] = 0
        surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255

        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()


def get_entry_point():
    return "InterfuserAgent"


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(input_size, need_scale=True,
                               mean=IMAGENET_DEFAULT_MEAN,
                               std=IMAGENET_DEFAULT_STD):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []
    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size
    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean),
                                    std=torch.tensor(std)))
    return transforms.Compose(tfl)


# ----------------------------------------------------------------------
# InterfuserAgent
# ----------------------------------------------------------------------
class InterfuserAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self._hic = DisplayInterface()
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.tracker = Tracker()
        self.instance_tracker = InstanceTracker(max_disappeared=5, max_distance=100)
        print("✓ Instance tracker initialized")

        # Digital Twin Config (same-machine default)
        twin_host = os.environ.get("DIGITAL_TWIN_HOST", "localhost")
        twin_port = int(os.environ.get("DIGITAL_TWIN_PORT", "9999"))
        twin_enabled = os.environ.get("DIGITAL_TWIN_ENABLED", "true").lower() == "true"

        self.twin_broadcaster = DigitalTwinBroadcaster(
            twin_host=twin_host,
            twin_port=twin_port,
            enabled=twin_enabled
        )

        if twin_enabled and self.twin_broadcaster.connected:
            print("✓ Digital Twin connection established (same machine mode)")

        self.input_buffer = {
            "rgb": deque(), "rgb_left": deque(), "rgb_right": deque(),
            "rgb_rear": deque(), "lidar": deque(),
            "gps": deque(), "thetas": deque(),
        }

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
        self.skip_frames = self.config.skip_frames
        self.controller = InterfuserController(self.config)

        if isinstance(self.config.model, list):
            self.ensemble = True
        else:
            self.ensemble = False

        if self.ensemble:
            for i in range(len(self.config.model)):
                self.nets = []
                net = create_model(self.config.model[i])
                path_to_model_file = self.config.model_path[i]
                print('load model: %s' % path_to_model_file)
                net.load_state_dict(torch.load(path_to_model_file)["state_dict"])
                net.cuda()
                net.eval()
                self.nets.append(net)
        else:
            self.net = create_model(self.config.model)
            path_to_model_file = self.config.model_path
            print('load model: %s' % path_to_model_file)
            self.net.load_state_dict(torch.load(path_to_model_file)["state_dict"])
            self.net.cuda()
            self.net.eval()

        self.softmax = torch.nn.Softmax(dim=1)
        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_control = None
        self.prev_surround_map = None

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second))
            )
            print(string)
            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    # ------------------------------------------------------------------
    # Sensors: add BEV semantic cam
    # ------------------------------------------------------------------
    def sensors(self):
        return [
            # RGB cameras
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "width": 800, "height": 600, "fov": 100, "id": "rgb"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": -60.0,
             "width": 400, "height": 300, "fov": 100, "id": "rgb_left"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 60.0,
             "width": 400, "height": 300, "fov": 100, "id": "rgb_right"},

            # Front-view semantic cams
            {"type": "sensor.camera.semantic_segmentation", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "width": 800, "height": 600, "fov": 100, "id": "seg_front"},
            {"type": "sensor.camera.semantic_segmentation", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": -60.0,
             "width": 400, "height": 300, "fov": 100, "id": "seg_left"},
            {"type": "sensor.camera.semantic_segmentation", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 60.0,
             "width": 400, "height": 300, "fov": 100, "id": "seg_right"},

            # NEW: BEV semantic cam (top-down)
            {"type": "sensor.camera.semantic_segmentation",
             "x": 0.0, "y": 0.0, "z": 40.0,
             "roll": 0.0, "pitch": -90.0, "yaw": 0.0,
             "width": 512, "height": 512, "fov": 60,
             "id": "bev_sem"},

            # LiDAR + IMU + GPS + speedometer
            {"type": "sensor.lidar.ray_cast", "x": 1.3, "y": 0.0, "z": 2.5,
             "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "id": "lidar"},
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": 0.05, "id": "imu"},
            {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": 0.01, "id": "gps"},
            {"type": "sensor.speedometer",
             "reading_frequency": 20, "id": "speed"},
        ]

    # ------------------------------------------------------------------
    # Tick: process sensors, including BEV semantics
    # ------------------------------------------------------------------
    def tick(self, input_data):
        # RGB
        rgb = cv2.cvtColor(input_data["rgb"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB)

        # Front semantic cameras
        seg_front = input_data["seg_front"][1][:, :, :3]
        seg_left = input_data["seg_left"][1][:, :, :3]
        seg_right = input_data["seg_right"][1][:, :, :3]

        seg_front_colored = convert_segmentation_to_rgb(seg_front)
        seg_left_colored = convert_segmentation_to_rgb(seg_left)
        seg_right_colored = convert_segmentation_to_rgb(seg_right)

        inst_front, inst_map = create_pseudo_instances_from_semantic(seg_front)
        inst_left, _ = create_pseudo_instances_from_semantic(seg_left)
        inst_right, _ = create_pseudo_instances_from_semantic(seg_right)

        inst_front_colored = convert_instance_to_rgb(inst_front)
        inst_left_colored = convert_instance_to_rgb(inst_left)
        inst_right_colored = convert_instance_to_rgb(inst_right)

        instance_analysis = analyze_instances(inst_front, seg_front)
        rgb_with_boxes = draw_instance_boxes(rgb, instance_analysis)

        detections = []
        for instance in instance_analysis['instances']:
            detections.append({
                'instance_id': instance['instance_id'],
                'class_name': instance['class_name'],
                'center': [instance['center_x'], instance['center_y']],
                'bbox': instance['bbox'],
                'additional_info': {
                    'pixel_count': instance['pixel_count'],
                    'class_id': instance['class_id']
                }
            })

        tracking_results = self.instance_tracker.update(detections)
        rgb_with_tracks = draw_tracked_objects(rgb, tracking_results, show_trails=True)

        # --- NEW: BEV semantic & instances ---
        bev_sem = input_data["bev_sem"][1][:, :, :3]         # BGR channels from CARLA
        bev_sem_colored = convert_segmentation_to_rgb(bev_sem)
        inst_bev, _ = create_pseudo_instances_from_semantic(bev_sem)
        bev_instance_analysis = analyze_instances(inst_bev, bev_sem)
        bev_sem_with_boxes = draw_instance_boxes(bev_sem_colored, bev_instance_analysis)
        bev_sem_with_boxes = cv2.resize(
            bev_sem_with_boxes, (400, 400),
            interpolation=cv2.INTER_NEAREST
        )
        # --------------------------------------

        # Vehicle state
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if math.isnan(compass):
            compass = 0.0

        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "rgb_with_boxes": rgb_with_boxes,
            "rgb_with_tracks": rgb_with_tracks,

            "seg_front": seg_front,
            "seg_left": seg_left,
            "seg_right": seg_right,
            "seg_front_colored": seg_front_colored,
            "seg_left_colored": seg_left_colored,
            "seg_right_colored": seg_right_colored,

            "inst_front": inst_front,
            "inst_left": inst_left,
            "inst_right": inst_right,
            "inst_front_colored": inst_front_colored,
            "inst_left_colored": inst_left_colored,
            "inst_right_colored": inst_right_colored,

            "instance_analysis": instance_analysis,
            "tracking_results": tracking_results,

            "gps": gps,
            "speed": speed,
            "compass": compass,

            # BEV semantic outputs
            "bev_sem": bev_sem,
            "inst_bev": inst_bev,
            "bev_instance_analysis": bev_instance_analysis,
            "bev_sem_with_boxes": bev_sem_with_boxes,
        }

        pos = self._get_position(result)

        # LiDAR → BEV histogram features
        lidar_data = input_data['lidar'][1]
        result['raw_lidar'] = lidar_data
        lidar_unprocessed = lidar_data[:, :3]
        lidar_unprocessed[:, 1] *= -1

        full_lidar = transform_2d_points(
            lidar_unprocessed,
            np.pi / 2 - compass, -pos[0], -pos[1],
            np.pi / 2 - compass, -pos[0], -pos[1]
        )
        lidar_processed = lidar_to_histogram_features(full_lidar, crop=224)

        if self.step % 2 == 0 or self.step < 4:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar
        result["gps"] = pos

        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_command"] = next_cmd.value

        result['measurements'] = [pos[0], pos[1], compass, speed]
        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        local_command_point = np.array(
            [next_wp[0] - pos[0], next_wp[1] - pos[1]]
        )
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point

        # Save frames (front-view dataset)
        rgb_frame = cv2.cvtColor(result["rgb"], cv2.COLOR_RGB2BGR)
        rgb_left_frame = cv2.cvtColor(result["rgb_left"], cv2.COLOR_RGB2BGR)
        rgb_right_frame = cv2.cvtColor(result["rgb_right"], cv2.COLOR_RGB2BGR)
        rgb_with_tracks_frame = cv2.cvtColor(
            result["rgb_with_tracks"], cv2.COLOR_RGB2BGR
        )

        seg_front_frame = result["seg_front"]
        seg_front_colored_frame = cv2.cvtColor(
            result["seg_front_colored"], cv2.COLOR_RGB2BGR
        )
        seg_left_frame = result["seg_left"]
        seg_right_frame = result["seg_right"]

        inst_front_frame = result["inst_front"]
        inst_front_colored_frame = cv2.cvtColor(
            result["inst_front_colored"], cv2.COLOR_RGB2BGR
        )
        inst_left_frame = result["inst_left"]
        inst_right_frame = result["inst_right"]

        threads = [
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["rgb"],
                                   f"{self.step:06d}.jpg"), rgb_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["rgb_left"],
                                   f"{self.step:06d}.jpg"), rgb_left_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["rgb_right"],
                                   f"{self.step:06d}.jpg"), rgb_right_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["seg_front"],
                                   f"{self.step:06d}.png"), seg_front_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["seg_left"],
                                   f"{self.step:06d}.png"), seg_left_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["seg_right"],
                                   f"{self.step:06d}.png"), seg_right_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["seg_colored_front"],
                                   f"{self.step:06d}.png"), seg_front_colored_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["inst_front"],
                                   f"{self.step:06d}.png"), inst_front_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["inst_left"],
                                   f"{self.step:06d}.png"), inst_left_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["inst_right"],
                                   f"{self.step:06d}.png"), inst_right_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["inst_colored_front"],
                                   f"{self.step:06d}.png"), inst_front_colored_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["tracking_visualization"],
                                   f"{self.step:06d}.jpg"), rgb_with_tracks_frame)
            ),
        ]

        for t in threads:
            t.start()

        analysis_path = os.path.join(
            FRAME_DIRS["inst_analysis"], f"{self.step:06d}.json"
        )
        with open(analysis_path, 'w') as f:
            json.dump(result["instance_analysis"], f, indent=2)

        if tracking_results:
            tracking_path = os.path.join(
                FRAME_DIRS["tracking"], f"{self.step:06d}.json"
            )
            tracking_export = {
                'frame': self.step,
                'timestamp': input_data['rgb'][0]
                if 'rgb' in input_data else 0.0,
                'tracked_objects': {
                    str(track_id): {
                        'class_name': info['class_name'],
                        'center': info['center'],
                        'bbox': info['bbox'],
                        'velocity': info['velocity'],
                        'frames_tracked': info['frames_tracked'],
                        'first_seen': info['first_seen']
                    }
                    for track_id, info in tracking_results.items()
                }
            }
            with open(tracking_path, 'w') as f:
                json.dump(tracking_export, f, indent=2)

        save_sensor_data_to_disk(self.step, result, input_data, FRAME_DIRS)

        return result

    # ------------------------------------------------------------------
    # Run step: InterFuser forward + BEV display + streaming
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
            return self.prev_control

        tick_data = self.tick(input_data)

        velocity = tick_data["speed"]
        command = tick_data["next_command"]

        rgb = self.rgb_front_transform(
            Image.fromarray(tick_data["rgb"])
        ).unsqueeze(0).cuda().float()
        rgb_left = self.rgb_left_transform(
            Image.fromarray(tick_data["rgb_left"])
        ).unsqueeze(0).cuda().float()
        rgb_right = self.rgb_right_transform(
            Image.fromarray(tick_data["rgb_right"])
        ).unsqueeze(0).cuda().float()
        rgb_center = self.rgb_center_transform(
            Image.fromarray(tick_data["rgb"])
        ).unsqueeze(0).cuda().float()

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = command - 1
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(velocity)
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float().unsqueeze(0).cuda()

        input_data_model = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "rgb_center": rgb_center,
            "measurements": mes,
            "target_point": torch.from_numpy(
                tick_data["target_point"]
            ).float().cuda().view(1, -1),
            "lidar": torch.from_numpy(
                tick_data["lidar"]
            ).float().cuda().unsqueeze(0)
        }

        if self.ensemble:
            outputs = []
            with torch.no_grad():
                for net in self.nets:
                    output = net(input_data_model)
                    outputs.append(output)
            traffic_meta = torch.mean(torch.stack([x[0] for x in outputs]), 0)
            pred_waypoints = torch.mean(torch.stack([x[1] for x in outputs]), 0)
            is_junction = torch.mean(torch.stack([x[2] for x in outputs]), 0)
            traffic_light_state = torch.mean(torch.stack([x[3] for x in outputs]), 0)
            stop_sign = torch.mean(torch.stack([x[4] for x in outputs]), 0)
            bev_feature = torch.mean(torch.stack([x[5] for x in outputs]), 0)
        else:
            with torch.no_grad():
                (
                    traffic_meta,
                    pred_waypoints,
                    is_junction,
                    traffic_light_state,
                    stop_sign,
                    bev_feature,
                ) = self.net(input_data_model)

        traffic_meta = traffic_meta.detach().cpu().numpy()[0]
        bev_feature = bev_feature.detach().cpu().numpy()[0]
        pred_waypoints = pred_waypoints.detach().cpu().numpy()[0]
        is_junction = self.softmax(is_junction).detach().cpu().numpy().reshape(-1)[0]
        traffic_light_state = self.softmax(
            traffic_light_state
        ).detach().cpu().numpy().reshape(-1)[0]
        stop_sign = self.softmax(
            stop_sign
        ).detach().cpu().numpy().reshape(-1)[0]

        if self.step % 2 == 0 or self.step < 4:
            traffic_meta = self.tracker.update_and_predict(
                traffic_meta.reshape(20, 20, -1),
                tick_data['gps'],
                tick_data['compass'],
                self.step // 2
            )
            traffic_meta = traffic_meta.reshape(400, -1)
            self.traffic_meta_moving_avg = (
                self.momentum * self.traffic_meta_moving_avg
                + (1 - self.momentum) * traffic_meta
            )

        traffic_meta = self.traffic_meta_moving_avg
        tick_data["raw"] = traffic_meta
        tick_data["bev_feature"] = bev_feature

        steer, throttle, brake, meta_infos = self.controller.run_step(
            velocity,
            pred_waypoints,
            is_junction,
            traffic_light_state,
            stop_sign,
            self.traffic_meta_moving_avg,
        )

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        surround_map, box_info = render(
            traffic_meta.reshape(20, 20, 7), pixels_per_meter=20
        )
        surround_map = surround_map[:400, 160:560]
        surround_map = np.stack([surround_map, surround_map, surround_map], 2)

        self_car_map = render_self_car(
            loc=np.array([0, 0]),
            ori=np.array([0, -1]),
            box=np.array([2.45, 1.0]),
            color=[1, 1, 0],
            pixels_per_meter=20
        )[:400, 160:560]

        pred_waypoints = pred_waypoints.reshape(-1, 2)
        safe_index = 10
        for i in range(10):
            if pred_waypoints[i, 0]**2 + pred_waypoints[i, 1]**2 > (meta_infos[3] + 0.5)**2:
                safe_index = i
                break

        wp1 = render_waypoints(
            pred_waypoints[:safe_index], pixels_per_meter=20,
            color=(0, 255, 0)
        )[:400, 160:560]
        wp2 = render_waypoints(
            pred_waypoints[safe_index:], pixels_per_meter=20,
            color=(255, 0, 0)
        )[:400, 160:560]
        wp = wp1 + wp2

        surround_map = np.clip(
            surround_map.astype(np.float32)
            + self_car_map.astype(np.float32)
            + wp.astype(np.float32),
            0, 255
        ).astype(np.uint8)

        map_t1, box_info = render(
            traffic_meta.reshape(20, 20, 7),
            pixels_per_meter=20, t=1
        )
        map_t1 = map_t1[:400, 160:560]
        map_t1 = np.stack([map_t1, map_t1, map_t1], 2)
        map_t1 = np.clip(
            map_t1.astype(np.float32) + self_car_map.astype(np.float32),
            0, 255
        ).astype(np.uint8)
        map_t1 = cv2.resize(map_t1, (200, 200))

        map_t2, box_info = render(
            traffic_meta.reshape(20, 20, 7),
            pixels_per_meter=20, t=2
        )
        map_t2 = map_t2[:400, 160:560]
        map_t2 = np.stack([map_t2, map_t2, map_t2], 2)
        map_t2 = np.clip(
            map_t2.astype(np.float32) + self_car_map.astype(np.float32),
            0, 255
        ).astype(np.uint8)
        map_t2 = cv2.resize(map_t2, (200, 200))

        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control
            self.prev_surround_map = surround_map

        # FIXED: Always use waypoint-based surround_map for local display (System A)
        # BEV semantic HD map is ONLY streamed to digital twin (System B)
        tick_data["map"] = self.prev_surround_map

        tick_data["map_t1"] = map_t1
        tick_data["map_t2"] = map_t2
        tick_data["rgb_raw"] = tick_data["rgb"]
        tick_data["rgb_left_raw"] = tick_data["rgb_left"]
        tick_data["rgb_right_raw"] = tick_data["rgb_right"]

        tick_data["rgb"] = cv2.resize(tick_data["rgb"], (800, 600))
        tick_data["rgb_left"] = cv2.resize(tick_data["rgb_left"], (200, 150))
        tick_data["rgb_right"] = cv2.resize(tick_data["rgb_right"], (200, 150))
        tick_data["rgb_focus"] = cv2.resize(
            tick_data["rgb_raw"][244:356, 344:456], (150, 150)
        )
        tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
            control.throttle,
            control.steer,
            control.brake,
        )
        tick_data["meta_infos"] = meta_infos
        tick_data["box_info"] = "car: %d, bike: %d, pedestrian: %d" % (
            box_info["car"],
            box_info["bike"],
            box_info["pedestrian"],
        )
        tick_data["mes"] = "speed: %.2f" % velocity
        tick_data["time"] = "time: %.3f" % timestamp

        surface = self._hic.run_interface(tick_data)
        tick_data["surface"] = surface

        # BROADCAST TO DIGITAL TWIN (including BEV HD map)
        if self.twin_broadcaster.enabled and self.twin_broadcaster.connected:
            try:
                tracking_results = tick_data.get('tracking_results', {}) or {}
                instance_counts = tick_data.get('instance_analysis', {}).get('counts', {})
                bev_img = tick_data.get('bev_sem_with_boxes')

                state_data = {
                    'timestamp': timestamp,
                    'step': self.step,
                    'vehicle_state': {
                        'position': tick_data['gps'].tolist()
                        if hasattr(tick_data['gps'], 'tolist')
                        else list(tick_data['gps']),
                        'velocity': float(tick_data['speed']),
                        'compass': float(tick_data['compass']),
                        'target_point': tick_data['target_point'].tolist()
                        if hasattr(tick_data['target_point'], 'tolist')
                        else list(tick_data['target_point']),
                        'next_command': int(tick_data['next_command']),
                    },
                    'control': {
                        'throttle': float(control.throttle),
                        'steer': float(control.steer),
                        'brake': float(control.brake),
                    },
                    'predictions': {
                        'is_junction': float(is_junction),
                        'traffic_light_state': float(traffic_light_state),
                        'stop_sign': float(stop_sign),
                    },
                    'tracking': {
                        'object_counts': instance_counts,
                        'num_tracked': len(tracking_results),
                        'tracked_objects': {
                            str(track_id): {
                                'class_name': info['class_name'],
                                'bbox': info['bbox'],
                                'center': info['center'],
                                'velocity': info['velocity'],
                                'frames_tracked': info['frames_tracked'],
                                'first_seen': info['first_seen'],
                            }
                            for track_id, info in tracking_results.items()
                        }
                    },
                    'images': {
                        'rgb': compress_image(tick_data['rgb_raw']),
                        'rgb_with_tracks': compress_image(tick_data['rgb_with_tracks']),
                        'seg_colored': compress_image(tick_data['seg_front_colored']),
                        'inst_colored': compress_image(tick_data['inst_front_colored']),
                        'bev_sem_with_boxes': compress_image(bev_img)
                        if bev_img is not None else None,
                    }
                }

                self.twin_broadcaster.send_state(state_data)
            except Exception as e:
                print(f"Error preparing twin data: {e}")

        if SAVE_PATH is not None:
            self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = self.step // self.skip_frames
        Image.fromarray(tick_data["surface"]).save(
            self.save_path / "meta" / ("%04d.jpg" % frame)
        )
        return

    def destroy(self):
        if hasattr(self, 'instance_tracker'):
            trajectories_path = os.path.join(
                DATASET_DIR, "complete_trajectories.json"
            )
            self.instance_tracker.export_tracks_to_json(trajectories_path)
            print(f"✓ Exported {len(self.instance_tracker.objects)} tracked object trajectories")

        if hasattr(self, 'twin_broadcaster'):
            self.twin_broadcaster.close()
            print("✓ Digital Twin connection closed")

        if self.ensemble:
            del self.nets
        else:
            del self.net

        dataset_info = {
            'total_frames': self.step + 1,
            'dataset_path': DATASET_DIR,
            'sensors': list(FRAME_DIRS.keys()),
            'collection_complete': True,
            'instance_type': 'pseudo_from_semantic',
        }

        summary_path = os.path.join(DATASET_DIR, 'dataset_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print(f"\n✓ Dataset collection complete! Total frames: {self.step + 1}")
