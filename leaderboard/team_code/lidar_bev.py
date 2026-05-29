import numpy as np
import cv2

def lidar_to_bev(lidar_points, ppm=5, radius=50):
    size = int(radius * 2 * ppm)
    bev = np.zeros((size, size), np.uint8)

    for x, y, z in lidar_points:
        if abs(x) < radius and abs(y) < radius:
            px = int(x * ppm + size // 2)
            py = int(-y * ppm + size // 2)
            bev[py, px] = min(255, bev[py, px] + 30)

    return cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)

