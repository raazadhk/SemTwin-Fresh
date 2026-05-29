# team_code/hd_vector_map.py

import numpy as np
import cv2
import carla


class HDVectorMap:
    """
    Simple vector-lane HD map for CARLA.

    - Precomputes lane polylines in world (x, y)
    - Each lane is a list of points
    - At runtime, we crop a local window around ego (x, y)
      and render it into a 2D BEV image.
    """

    def __init__(self, carla_map, pixels_per_meter=4, radius_m=40.0):
        """
        carla_map: carla.Map from world.get_map() or RoutePlanner._map
        pixels_per_meter: resolution of BEV image
        radius_m: half-size of local window in meters
        """
        self._map = carla_map
        self.ppm = pixels_per_meter
        self.radius = radius_m
        self.img_size = int(2 * radius_m * pixels_per_meter)
        self.center = self.img_size // 2

        # Precompute global lane polylines
        self._lanes = self._collect_lane_polylines()

    def _collect_lane_polylines(self):
        """
        Group waypoints by (road_id, lane_id) into polylines.
        Only uses Driving lanes.
        """
        waypoints = self._map.generate_waypoints(2.0)  # 2m spacing is enough
        lane_dict = {}

        for wp in waypoints:
            if wp.lane_type != carla.LaneType.Driving:
                continue
            key = (wp.road_id, wp.lane_id)
            loc = wp.transform.location
            lane_dict.setdefault(key, []).append((loc.x, loc.y))

        lane_polys = []
        for pts in lane_dict.values():
            pts = np.asarray(pts, dtype=np.float32)
            # Rough ordering along lane
            order = np.argsort(pts[:, 0] + pts[:, 1])
            pts = pts[order]
            lane_polys.append(pts)

        return lane_polys

    def render_local(self, ego_x, ego_y):
        """
        Render a local HD map window around (ego_x, ego_y) in world coords.

        Returns: uint8 image (H, W, 3) with lanes drawn.
        - Green: lane centerlines
        - Pink dot: ego position
        """
        img = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255  # white bg

        r2 = self.radius ** 2

        for poly in self._lanes:
            dx = poly[:, 0] - ego_x
            dy = poly[:, 1] - ego_y
            mask = (dx * dx + dy * dy) <= r2
            if mask.sum() < 2:
                continue

            local = np.stack([dx[mask], dy[mask]], axis=1)
            # world → image: x forward (up), y left/right
            px = (self.center + local[:, 0] * self.ppm).astype(np.int32)
            py = (self.center - local[:, 1] * self.ppm).astype(np.int32)
            pts = np.stack([px, py], axis=1)

            cv2.polylines(img, [pts], False, (0, 255, 0), 2)  # green lanes

        # ego marker
        cv2.circle(img, (self.center, self.center), 4, (255, 0, 255), -1)

        return img

