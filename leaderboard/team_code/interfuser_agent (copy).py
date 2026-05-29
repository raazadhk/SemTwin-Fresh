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

#DATASET_DIR = "interfuser_frame_dataset"
#FRAME_DIRS = {
#    "rgb": os.path.join(DATASET_DIR, "RGB"),
#    "rgb_left": os.path.join(DATASET_DIR, "RGB_Left"),
#    "rgb_right": os.path.join(DATASET_DIR, "RGB_Right"),
#}
#for dir_path in FRAME_DIRS.values():
#    os.makedirs(dir_path, exist_ok=True)

#Added portion 1
DATASET_DIR = "interfuser_complete_dataset"
FRAME_DIRS = {
    "rgb": os.path.join(DATASET_DIR, "RGB"),
    "rgb_left": os.path.join(DATASET_DIR, "RGB_Left"),
    "rgb_right": os.path.join(DATASET_DIR, "RGB_Right"),
    "lidar": os.path.join(DATASET_DIR, "LiDAR"),
    "measurements": os.path.join(DATASET_DIR, "Measurements"),
    "bev_features": os.path.join(DATASET_DIR, "BEV_Features"),
    "metadata": os.path.join(DATASET_DIR, "Metadata"),
}
for dir_path in FRAME_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)
#upto here for 1

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


def create_carla_rgb_transform(input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
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
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
    return transforms.Compose(tfl)
    
#Added portion 2
def save_sensor_data_to_disk(step, tick_data, input_data, frame_dirs):
    """
    Save all sensor data to disk
    Called from tick() method
    """
    frame_id = f"{step:06d}"
    
    # 1. LiDAR (save as .npy)
    if 'raw_lidar' in tick_data:
        lidar_path = os.path.join(frame_dirs['lidar'], f"{frame_id}.npy")
        np.save(lidar_path, tick_data['raw_lidar'])
    
    # 2. Measurements (GPS, IMU, Speed, etc.)
    measurements = {
        'gps': tick_data['gps'].tolist() if hasattr(tick_data['gps'], 'tolist') else list(tick_data['gps']),
        'compass': float(tick_data['compass']),
        'speed': float(tick_data['speed']),
        'measurements': tick_data['measurements'],  # [pos_x, pos_y, compass, speed]
        'target_point': tick_data['target_point'].tolist() if hasattr(tick_data['target_point'], 'tolist') else list(tick_data['target_point']),
        'next_command': int(tick_data['next_command']),
        'frame_id': frame_id,
        'timestamp': input_data['rgb'][0] if 'rgb' in input_data else 0.0
    }
    
    measurements_path = os.path.join(frame_dirs['measurements'], f"{frame_id}.json")
    with open(measurements_path, 'w') as f:
        json.dump(measurements, f, indent=2)    
#upto here for 2

class InterfuserAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        # -----------------------
        # Display Interface, etc
        # -----------------------
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
        self.input_buffer = {
            "rgb": deque(),
            "rgb_left": deque(),
            "rgb_right": deque(),
            "rgb_rear": deque(),
            "lidar": deque(),
            "gps": deque(),
            "thetas": deque(),
        }

        # -----------------------
        # Load Config and Model
        # -----------------------
        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
        self.skip_frames = self.config.skip_frames
        self.controller = InterfuserController(self.config)
        if isinstance(self.config.model, list):
            self.ensemble = True
        else:
            self.ensemble = False

        # Load single or ensemble model(s)
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

        # Softmax for certain classifier outputs
        self.softmax = torch.nn.Softmax(dim=1)
        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_control = None
        self.prev_surround_map = None

        # -----------------------
        # Save Path for logs
        # -----------------------
        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(map(lambda x: "%02d" % x, (now.month, now.day, now.hour, now.minute, now.second)))
            print(string)
            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

        # -----------------------
        # (NEW) Set up Attention Extraction
        # -----------------------
        self.attention_maps = {}
        os.makedirs("attention_images_1", exist_ok=True)  # Directory to save attention map images

        def get_attention_hook(name):
            """ Creates a forward hook that captures the layer's output into self.attention_maps[name]. """
            def hook(module, input, output):
                # Convert output to numpy for easier handling
                self.attention_maps[name] = output.detach().cpu().numpy()
            return hook

        # Example: If your model has a layer called 'self_attention'
        # You may need to adjust this to your actual attention module name(s).
        if hasattr(self.net, 'self_attention'):
            print("[INFO] Registering hook on self.net.self_attention")
            self.net.self_attention.register_forward_hook(get_attention_hook("self_attention"))
        elif hasattr(self.net, 'attn'):
            print("[INFO] Registering hook on self.net.attn")
            self.net.attn.register_forward_hook(get_attention_hook("attn"))
        # If your attention layers have different names, add additional checks here.

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "width": 800, "height": 600, "fov": 100, "id": "rgb"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": -60.0, "width": 400, "height": 300, "fov": 100, "id": "rgb_left"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 60.0, "width": 400, "height": 300, "fov": 100, "id": "rgb_right"},
            {"type": "sensor.lidar.ray_cast", "x": 1.3, "y": 0.0, "z": 2.5,
             "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "id": "lidar"},
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.05, "id": "imu"},
            {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.01, "id": "gps"},
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    #def reduce_image_quality(self, image, scale_percent=10):
        #width = int(image.shape[1] * scale_percent / 100)
        #height = int(image.shape[0] * scale_percent / 100)
        #dim = (width, height)
        #resized_down = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        #resized_original = cv2.resize(resized_down, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        #noise_factor = 10
        #noise = np.random.randint(0, noise_factor, resized_original.shape, dtype='uint8')
        #noisy_image = cv2.add(resized_original, noise)
        #_, compressed = cv2.imencode('.jpg', noisy_image, [cv2.IMWRITE_JPEG_QUALITY, 5])
        #compressed_image_now = cv2.imdecode(compressed, 1)
        #compressed_image = (compressed_image_now // 32) * 32
        #return compressed_image

    #def apply_random_blockades(self, image, blockade_size=(200, 200), number_of_blockades=20):
        #modified_image = image.copy()
        #for _ in range(number_of_blockades):
            #upper_left_x = np.random.randint(0, image.shape[1] - blockade_size[0])
            #upper_left_y = np.random.randint(0, image.shape[0] - blockade_size[1])
            #blockade = np.zeros((blockade_size[1], blockade_size[0], 3), dtype=np.uint8)
            #modified_image[upper_left_y:upper_left_y + blockade_size[1], upper_left_x:upper_left_x + blockade_size[0]] = blockade
        #return modified_image
     
    
    def tick(self, input_data):
        rgb = cv2.cvtColor(input_data["rgb"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        #rgb = self.apply_random_blockades(rgb_now)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        #rgb_left = self.reduce_image_quality(rgb_left_now, scale_percent=10)
        rgb_right = cv2.cvtColor(input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if math.isnan(compass):
            compass = 0.0

        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "gps": gps,
            "speed": speed,
            "compass": compass
        }

        pos = self._get_position(result)
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
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point

        # Save frames for a dataset
        rgb_frame = cv2.cvtColor(result["rgb"], cv2.COLOR_RGB2BGR)
        rgb_left_frame = cv2.cvtColor(result["rgb_left"], cv2.COLOR_RGB2BGR)
        rgb_right_frame = cv2.cvtColor(result["rgb_right"], cv2.COLOR_RGB2BGR)
        
        #cv2.imwrite(os.path.join(FRAME_DIRS["rgb"], f"{frame_count:06d}.jpg"), rgb_frame)
        #cv2.imwrite(os.path.join(FRAME_DIRS["rgb_left"], f"{frame_count:06d}.jpg"), rgb_left_frame)
        #cv2.imwrite(os.path.join(FRAME_DIRS["rgb_right"], f"{frame_count:06d}.jpg"), rgb_right_frame)


        threads = [
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["rgb"], f"{self.step:06d}.jpg"), rgb_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["rgb_left"], f"{self.step:06d}.jpg"), rgb_left_frame)
            ),
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(FRAME_DIRS["rgb_right"], f"{self.step:06d}.jpg"), rgb_right_frame)
            )
        ]

        for t in threads:
            t.start()
        
        # Added portion 3
        # Save additional sensor data (LiDAR, measurements)
        save_sensor_data_to_disk(self.step, result, input_data, FRAME_DIRS)
        # end here for 3
        
        return result


    #def send_frame_to_multiple_jetsons(self, frame, jetson_devices):
        #for camera, (ip, port) in jetson_devices.items():
            #client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #client_socket.connect((ip, port))
            
            # Serialize the frame
            #data = pickle.dumps(frame[camera])
            
            # Send the size of the serialized data
            #data_size = len(data)
            #client_socket.sendall(struct.pack('!I', data_size))
            
            # Send the actual serialized data
            #client_socket.sendall(data)
            
            #client_socket.close()
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
            return self.prev_control

        tick_data = self.tick(input_data)
        
        # Create frame data for each camera
        #frames = {
            #'rgb': {'frame': tick_data['rgb'], 'timestamp': timestamp},
            #'rgb_left': {'frame': tick_data['rgb_left'], 'timestamp': timestamp},
            #'rgb_right': {'frame': tick_data['rgb_right'], 'timestamp': timestamp}
        #}

        # List of Jetson device IP addresses and ports for each camera
        #jetson_devices = {
            #'rgb': ('192.168.0.122', 12345),
            #'rgb_left': ('192.168.0.204', 12346),
            #'rgb_right': ('192.168.0.166', 12347)  
        #}
        
        # Send frame to multiple Jetsons
        #self.send_frame_to_multiple_jetsons(frames, jetson_devices)
        
        velocity = tick_data["speed"]
        command = tick_data["next_command"]

        # Prepare model inputs
        rgb = self.rgb_front_transform(Image.fromarray(tick_data["rgb"])).unsqueeze(0).cuda().float()
        rgb_left = self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"])).unsqueeze(0).cuda().float()
        rgb_right = self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"])).unsqueeze(0).cuda().float()
        rgb_center = self.rgb_center_transform(Image.fromarray(tick_data["rgb"])).unsqueeze(0).cuda().float()

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = command - 1
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(velocity)
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float().unsqueeze(0).cuda()

        input_data = {}
        input_data["rgb"] = rgb
        input_data["rgb_left"] = rgb_left
        input_data["rgb_right"] = rgb_right
        input_data["rgb_center"] = rgb_center
        input_data["measurements"] = mes
        input_data["target_point"] = (
            torch.from_numpy(tick_data["target_point"]).float().cuda().view(1, -1)
        )
        input_data["lidar"] = (
            torch.from_numpy(tick_data["lidar"]).float().cuda().unsqueeze(0)
        )

        # Inference
        if self.ensemble:
            outputs = []
            with torch.no_grad():
                for net in self.nets:
                    output = net(input_data)
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
                ) = self.net(input_data)
        traffic_meta = traffic_meta.detach().cpu().numpy()[0]
        bev_feature = bev_feature.detach().cpu().numpy()[0]
        pred_waypoints = pred_waypoints.detach().cpu().numpy()[0]
        is_junction = self.softmax(is_junction).detach().cpu().numpy().reshape(-1)[0]
        traffic_light_state = (
            self.softmax(traffic_light_state).detach().cpu().numpy().reshape(-1)[0]
        )
        stop_sign = self.softmax(stop_sign).detach().cpu().numpy().reshape(-1)[0]

        if self.step % 2 == 0 or self.step < 4:
            traffic_meta = self.tracker.update_and_predict(
                traffic_meta.reshape(20, 20, -1), tick_data['gps'],
                tick_data['compass'], self.step // 2
            )
            traffic_meta = traffic_meta.reshape(400, -1)
            self.traffic_meta_moving_avg = (
                self.momentum * self.traffic_meta_moving_avg
                + (1 - self.momentum) * traffic_meta
            )

        traffic_meta = self.traffic_meta_moving_avg
        tick_data["raw"] = traffic_meta
        tick_data["bev_feature"] = bev_feature
        
        # Added portion 4
        # Save BEV features and model predictions (optional but useful)
		#if self.step % self.skip_frames == 0:  
		#    frame_id = f"{self.step:06d}"
		#    
		#    # Save BEV feature
		#    bev_path = os.path.join(FRAME_DIRS['bev_features'], f"{frame_id}.npy")
		#    np.save(bev_path, bev_feature)
		#    
		#    # Save additional metadata
		#    metadata = {
		#        'frame_id': frame_id,
		#        'timestamp': timestamp,
		#        'is_junction': float(is_junction),
		#        'traffic_light_state': float(traffic_light_state),
		#        'stop_sign': float(stop_sign),
		#        'pred_waypoints': pred_waypoints.tolist(),
		#        'velocity': float(velocity),
		#        'command': int(command)
		#    }
		#    
		#    metadata_path = os.path.join(FRAME_DIRS['metadata'], f"{frame_id}.json")
		#    with open(metadata_path, 'w') as f:
		#        json.dump(metadata, f, indent=2)
		#        
		#   # end of the 4      

        
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

        surround_map, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20)
        surround_map = surround_map[:400, 160:560]
        surround_map = np.stack([surround_map, surround_map, surround_map], 2)
        self_car_map = render_self_car(
            loc=np.array([0, 0]), ori=np.array([0, -1]),
            box=np.array([2.45, 1.0]), color=[1, 1, 0], pixels_per_meter=20
        )[:400, 160:560]

        pred_waypoints = pred_waypoints.reshape(-1, 2)
        safe_index = 10
        for i in range(10):
            if pred_waypoints[i, 0]**2 + pred_waypoints[i, 1]**2 > (meta_infos[3] + 0.5)**2:
                safe_index = i
                break

        wp1 = render_waypoints(pred_waypoints[:safe_index], pixels_per_meter=20, color=(0, 255, 0))[:400, 160:560]
        wp2 = render_waypoints(pred_waypoints[safe_index:], pixels_per_meter=20, color=(255, 0, 0))[:400, 160:560]
        wp = wp1 + wp2
        surround_map = np.clip(
            (surround_map.astype(np.float32) +
             self_car_map.astype(np.float32) +
             wp.astype(np.float32)),
            0, 255
        ).astype(np.uint8)

        map_t1, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=1)
        map_t1 = map_t1[:400, 160:560]
        map_t1 = np.stack([map_t1, map_t1, map_t1], 2)
        map_t1 = np.clip(map_t1.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
        map_t1 = cv2.resize(map_t1, (200, 200))

        map_t2, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=2)
        map_t2 = map_t2[:400, 160:560]
        map_t2 = np.stack([map_t2, map_t2, map_t2], 2)
        map_t2 = np.clip(map_t2.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
        map_t2 = cv2.resize(map_t2, (200, 200))

        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control
            self.prev_surround_map = surround_map

        tick_data["map"] = self.prev_surround_map
        tick_data["map_t1"] = map_t1
        tick_data["map_t2"] = map_t2
        tick_data["rgb_raw"] = tick_data["rgb"]
        tick_data["rgb_left_raw"] = tick_data["rgb_left"]
        tick_data["rgb_right_raw"] = tick_data["rgb_right"]

        tick_data["rgb"] = cv2.resize(tick_data["rgb"], (800, 600))
        tick_data["rgb_left"] = cv2.resize(tick_data["rgb_left"], (200, 150))
        tick_data["rgb_right"] = cv2.resize(tick_data["rgb_right"], (200, 150))
        tick_data["rgb_focus"] = cv2.resize(tick_data["rgb_raw"][244:356, 344:456], (150, 150))
        tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (control.throttle, control.steer, control.brake)
        tick_data["meta_infos"] = meta_infos
        tick_data["box_info"] = "car: %d, bike: %d, pedestrian: %d" % (box_info["car"], box_info["bike"], box_info["pedestrian"])
        tick_data["mes"] = "speed: %.2f" % velocity
        tick_data["time"] = "time: %.3f" % timestamp

        surface = self._hic.run_interface(tick_data)
        tick_data["surface"] = surface

        # (NEW) Convert and Save Attention Maps (if any)
        # -----------------------------------------------------
        if self.attention_maps:
            for layer_name, attn_map in self.attention_maps.items():
                # attn_map shape might be (B, Heads, H, W) or something else, depending on your model
                # For demonstration, let's assume (1, nHeads, h, w).
                # We'll average across the 'Heads' dimension to get a single 2D map.
                if len(attn_map.shape) >= 3:
                    # Example: average across dimension 1 (heads)
                    attn_map_2d = np.mean(attn_map, axis=1)[0]  # [0] -> first batch index
                else:
                    # If it's already 2D or 3D, adapt as needed
                    attn_map_2d = attn_map[0]

                # Normalize to [0, 1]
                amin, amax = attn_map_2d.min(), attn_map_2d.max()
                if amax - amin > 1e-6:
                    attn_map_2d = (attn_map_2d - amin) / (amax - amin)
                else:
                    attn_map_2d = np.zeros_like(attn_map_2d)

                # Resize to match the main camera's input resolution (here, 224x224) or (800x600)
                attn_map_2d = cv2.resize(attn_map_2d, (224, 224))

                # Convert to heatmap
                heatmap = cv2.applyColorMap((attn_map_2d * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # Save as an image
                cv2.imwrite(f"attention_images/{layer_name}_step{self.step}.png", heatmap)

        # -----------------------------------------------------

        if SAVE_PATH is not None:
            self.save(tick_data)
        return control

    def save(self, tick_data):
        frame = self.step // self.skip_frames
        Image.fromarray(tick_data["surface"]).save(self.save_path / "meta" / ("%04d.jpg" % frame))
        return

    def destroy(self):
        if self.ensemble:
            del self.nets
        else:
            del self.net
            
        #Added portion 5
        # Create dataset summary
	dataset_info = {
		    'total_frames': self.step + 1,
		    'dataset_path': DATASET_DIR,
		    'sensors': list(FRAME_DIRS.keys()),
		    'collection_complete': True
	}
		
	summary_path = os.path.join(DATASET_DIR, 'dataset_summary.json')
	with open(summary_path, 'w') as f:
		   json.dump(dataset_info, f, indent=2)
		
	print(f"\n✓ Dataset collection complete!")
	print(f"  Total frames: {self.step + 1}")
	print(f"  Location: {DATASET_DIR}")
	print(f"  Sensors collected: {len(FRAME_DIRS)}")
        #end to 5

