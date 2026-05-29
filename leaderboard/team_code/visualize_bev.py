import numpy as np
import cv2

def render_occupancy(occ):
    img = np.zeros((*occ.shape, 3), dtype=np.uint8)
    img[occ > 3] = (255, 255, 255)
    return img

def show_bev(win, img):
    cv2.imshow(win, cv2.resize(img, (400, 400)))
    cv2.waitKey(1)

