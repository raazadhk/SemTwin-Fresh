import numpy as np
import cv2

def extract_road_edges(occupancy, hit_threshold=2):
    """
    occupancy: [H,W] numpy array with hit counts
    returns: binary edge map [H,W]
    """
    occ_img = (occupancy > hit_threshold).astype(np.uint8) * 255
    if occ_img.max() == 0:
        return np.zeros_like(occ_img)
    edges = cv2.Canny(occ_img, 50, 150)
    return edges

def infer_lane_centerline(ego_history):
    if len(ego_history) < 10:
        return None
    xs = [p[0] for p in ego_history]
    ys = [p[1] for p in ego_history]
    return np.polyfit(xs, ys, deg=2)
