import carla
import numpy as np
import cv2

class HDMapRaster:
    def __init__(self, world, ppm=5, radius=50):
        self.map = world.get_map()
        self.ppm = ppm
        self.radius = radius
        self.size = int(radius * 2 * ppm)

        self.global_img = self._rasterize_global_map()

    def _rasterize_global_map(self):
        img = np.zeros((4000, 4000, 3), np.uint8)
        waypoints = self.map.generate_waypoints(1.0)

        for wp in waypoints:
            loc = wp.transform.location
            x = int(loc.x * self.ppm + 2000)
            y = int(loc.y * self.ppm + 2000)
            cv2.circle(img, (x, y), 1, (80, 80, 80), -1)

        return img

    def crop(self, ego_loc):
        cx = int(ego_loc.x * self.ppm + 2000)
        cy = int(ego_loc.y * self.ppm + 2000)
        r = self.size // 2
        return self.global_img[cy-r:cy+r, cx-r:cx+r].copy()

