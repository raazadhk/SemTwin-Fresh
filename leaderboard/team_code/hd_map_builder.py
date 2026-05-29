import torch
import numpy as np
import math
import torch.nn.functional as F

class HDMapBuilder:
    """
    Training-free online HD map builder using LiDAR + ego motion.
    Works in a local 'world' frame defined by the first ego pose.
    """

    def __init__(self, map_size_m=120.0, resolution=0.25, device="cpu"):
        self.map_size_m = map_size_m
        self.resolution = resolution
        self.device = device

        self.size_px = int(map_size_m / resolution)
        self.H = self.size_px
        self.W = self.size_px

        # Simple occupancy (hit count) map
        self.occupancy = torch.zeros(
            (1, self.H, self.W), dtype=torch.float32, device=device
        )
        self.confidence = torch.zeros_like(self.occupancy)

        # Origin (world-frame anchor)
        self.origin_xy = None

    def _init_origin(self, ego_xy):
        if self.origin_xy is None:
            self.origin_xy = torch.tensor(ego_xy, dtype=torch.float32, device=self.device)

    def _world_to_map(self, xy_world):
        """
        xy_world: [N,2] tensor
        returns: [N,2] of (i,j) float indices
        """
        d = xy_world - self.origin_xy  # relative to origin
        px = d[:, 0] / self.resolution + self.W / 2.0
        py = -d[:, 1] / self.resolution + self.H / 2.0
        return torch.stack([py, px], dim=-1)

    def update_from_ego(self, lidar_ego, ego_xy, ego_yaw):
        """
        lidar_ego : [N,3] points in ego frame (x forward, y left/right)
        ego_xy    : (x,y) in local 'world' frame (we use tick_data['gps'])
        ego_yaw   : heading (radians), we use tick_data['compass']
        """
        if isinstance(lidar_ego, np.ndarray):
            lidar_ego = torch.from_numpy(lidar_ego).float().to(self.device)

        ego_xy = torch.tensor(ego_xy, dtype=torch.float32, device=self.device)
        self._init_origin(ego_xy)

        pts_xy = lidar_ego[:, :2]  # [N,2] in ego frame

        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)
        R = torch.tensor([[cos_y, -sin_y], [sin_y, cos_y]],
                         dtype=torch.float32, device=self.device)  # 2x2

        pts_world = (R @ pts_xy.T).T
        pts_world[:, 0] += ego_xy[0]
        pts_world[:, 1] += ego_xy[1]

        ij = self._world_to_map(pts_world)
        i = ij[:, 0].long()
        j = ij[:, 1].long()

        valid = (i >= 0) & (i < self.H) & (j >= 0) & (j < self.W)
        i = i[valid]
        j = j[valid]

        self.occupancy[0, i, j] += 1.0
        self.confidence[0, i, j] += 1.0

    def get_local_map(self, ego_xy, size_m=60.0):
        """
        Returns local occupancy crop [1,Hc,Wc] around ego.
        """
        ego_xy = torch.tensor(ego_xy, dtype=torch.float32, device=self.device)
        self._init_origin(ego_xy)

        crop_px = int(size_m / self.resolution)
        half = crop_px // 2

        ego_ij = self._world_to_map(ego_xy.unsqueeze(0))[0]
        ci = int(round(ego_ij[0].item()))
        cj = int(round(ego_ij[1].item()))

        i0 = max(0, ci - half)
        i1 = min(self.H, ci + half)
        j0 = max(0, cj - half)
        j1 = min(self.W, cj + half)

        occ = self.occupancy[:, i0:i1, j0:j1]

        pad_t = max(0, half - (ci - i0))
        pad_b = max(0, (ci + half) - i1)
        pad_l = max(0, half - (cj - j0))
        pad_r = max(0, (cj + half) - j1)

        if pad_t or pad_b or pad_l or pad_r:
            occ = F.pad(occ, (pad_l, pad_r, pad_t, pad_b))

        return occ

    def get_drivable_mask(self, threshold=3):
        """
        Simple drivable estimate: low occupancy count → likely free.
        """
        return (self.occupancy < threshold).float()

