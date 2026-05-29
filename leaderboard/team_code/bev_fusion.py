import cv2
import numpy as np

def fuse_bev(hd_map, lidar_bev, interfuser_bev):
    fused = hd_map.copy()

    fused = cv2.addWeighted(fused, 1.0, lidar_bev, 0.8, 0)
    fused = cv2.addWeighted(fused, 1.0, interfuser_bev, 1.2, 0)

    return fused

