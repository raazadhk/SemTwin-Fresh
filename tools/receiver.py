import socket
import pickle
import struct
import cv2
import numpy as np
import math
import copy
import os
import logging
import threading
import time
from io import BytesIO
from PIL import Image
from datetime import datetime
from functools import partial
from collections import OrderedDict
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def receive_frame(port, save_dir, camera_type, frame_count_dict, interfuser_model):
    logging.info(f"Initializing {camera_type} receiver on port {port}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('', port))
    server_socket.listen(1)
    logging.info(f"Listening for {camera_type} frames on port {port}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    while True:
        conn, addr = server_socket.accept()
        data = b''
        payload_size = struct.calcsize('!I')
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack('!I', packed_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        frame = pickle.loads(frame_data)
        timestamp = frame['timestamp']
        img = frame['frame']
        Image.fromarray(img).save(os.path.join(save_dir, f'{timestamp}_{camera_type}.jpg'))
        analyze_frame(img, camera_type, timestamp, save_dir)
        frame_count_dict[camera_type] += 1
        x = {
            "rgb": img,
            "rgb_left": img,
            "rgb_right": img,
            "rgb_center": img,
            "measurements": torch.tensor([]),
            "target_point": torch.tensor([]),
            "lidar": torch.tensor([])
        }
        output = interfuser_model(x)
        conn.close()
        return  # gracefully exits after one cycle


def start_receiving(interfuser_model):
    logging.info("Starting multi-camera receiver pipeline...")
    ports_dirs = [
        (23456, "/home/carla100/analysis/rgb_frames", "rgb"),
        (23457, "/home/carla100/analysis/rgb_left_frames", "rgb_left"),
    ]
    frame_count_dict = {"rgb": 0, "rgb_left": 0}
    threads = []
    for port, save_dir, camera_type in ports_dirs:
        thread = threading.Thread(
            target=receive_frame,
            args=(port, save_dir, camera_type, frame_count_dict, interfuser_model)
        )
        threads.append(thread)
        thread.start()
    logging.info(f"Receiver threads initialized: {[t.name for t in threads]}")
    return  # hands off cleanly without blocking


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_attn_mask(mask_type):
    mask = torch.ones((151, 151), dtype=torch.bool).cuda()
    if mask_type == "seperate_all":
        mask[:50, :50] = False
        mask[50:67, 50:67] = False
        mask[67:84, 67:84] = False
        mask[84:101, 84:101] = False
        mask[101:151, 101:151] = False
    elif mask_type == "seperate_view":
        mask[:50, :50] = False
        mask[50:67, 50:67] = False
        mask[67:84, 67:84] = False
        mask[84:101, 84:101] = False
        mask[101:151, :] = False
        mask[:, 101:151] = False
    return mask


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=nn.ReLU, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=nn.ReLU, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(self.norm2(tgt), query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
        return tgt + self.dropout3(tgt2)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        return self.norm(output) if self.norm is not None else output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class GRUWaypointsPredictor(nn.Module):
    def __init__(self, input_dim, waypoints=10):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=64, batch_first=True)
        self.encoder = nn.Linear(2, 64)
        self.decoder = nn.Linear(64, 2)
        self.waypoints = waypoints

    def forward(self, x, target_point):
        bs = x.shape[0]
        z = self.encoder(target_point).unsqueeze(0)
        output, _ = self.gru(x, z)
        output = self.decoder(output.reshape(bs * self.waypoints, -1))
        return torch.cumsum(output.reshape(bs, self.waypoints, 2), 1)


class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format="NCHW"):
        super().__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature) if temperature else 1.0
        pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., height), np.linspace(-1., 1., width))
        self.register_buffer("pos_x", torch.from_numpy(pos_x.reshape(height * width)).float())
        self.register_buffer("pos_y", torch.from_numpy(pos_y.reshape(height * width)).float())

    def forward(self, feature):
        feature = feature.view(-1, self.height * self.width)
        weight = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * weight, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * weight, dim=1, keepdim=True)
        feature_keypoints = torch.cat([expected_x, expected_y], 1).view(-1, self.channel, 2)
        feature_keypoints[:, :, 1] = (feature_keypoints[:, :, 1] - 1) * 12
        feature_keypoints[:, :, 0] = feature_keypoints[:, :, 0] * 12
        return feature_keypoints


HOST = "0.0.0.0"   # Listen on all interfaces
PORT = 9999        # Must match DIGITAL_TWIN_PORT in the InterFuser agent


def recv_exact(sock, n_bytes):
    """Receive exactly n_bytes from the socket."""
    data = b""
    while len(data) < n_bytes:
        chunk = sock.recv(n_bytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def jpeg_bytes_to_bgr(img_bytes):
    """Convert JPEG bytes to an OpenCV BGR image."""
    if img_bytes is None:
        return None
    try:
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def draw_control_panel(img, throttle, steer, brake):
    """
    Draw a driving command panel on the bottom of the BEV image.
    Shows throttle (green bar), brake (red bar), and steer (centered blue bar).
    """
    h, w = img.shape[:2]

    # Panel background — semi-transparent dark strip at the bottom
    panel_h = 70
    panel_y = h - panel_h
    overlay = img.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    bar_max_w = 120   # max pixel width of each bar
    bar_h = 12        # bar height
    label_x = 8       # left margin for labels
    bar_x = 80        # x where bars start

    # ── Throttle ──────────────────────────────────────────────────────
    row_y = panel_y + 16
    cv2.putText(img, "THROT", (label_x, row_y + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    # Background track
    cv2.rectangle(img, (bar_x, row_y), (bar_x + bar_max_w, row_y + bar_h),
                  (60, 60, 60), -1)
    # Fill
    fill_w = int(throttle * bar_max_w)
    cv2.rectangle(img, (bar_x, row_y), (bar_x + fill_w, row_y + bar_h),
                  (0, 210, 60), -1)          # green
    # Value text
    cv2.putText(img, f"{throttle:.2f}", (bar_x + bar_max_w + 6, row_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 60), 1)

    # ── Brake ─────────────────────────────────────────────────────────
    row_y = panel_y + 36
    cv2.putText(img, "BRAKE", (label_x, row_y + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.rectangle(img, (bar_x, row_y), (bar_x + bar_max_w, row_y + bar_h),
                  (60, 60, 60), -1)
    fill_w = int(brake * bar_max_w)
    cv2.rectangle(img, (bar_x, row_y), (bar_x + fill_w, row_y + bar_h),
                  (0, 50, 220), -1)          # red (BGR)
    cv2.putText(img, f"{brake:.2f}", (bar_x + bar_max_w + 6, row_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 50, 220), 1)

    # ── Steer ─────────────────────────────────────────────────────────
    # Steer range [-1, 1] → centered bar; left = negative, right = positive
    row_y = panel_y + 56
    cv2.putText(img, "STEER", (label_x, row_y + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    mid_x = bar_x + bar_max_w // 2
    cv2.rectangle(img, (bar_x, row_y), (bar_x + bar_max_w, row_y + bar_h),
                  (60, 60, 60), -1)
    # Center tick
    cv2.line(img, (mid_x, row_y), (mid_x, row_y + bar_h), (150, 150, 150), 1)
    steer_fill = int(abs(steer) * (bar_max_w // 2))
    if steer >= 0:                            # turning right
        cv2.rectangle(img, (mid_x, row_y),
                      (mid_x + steer_fill, row_y + bar_h),
                      (255, 160, 0), -1)      # orange
    else:                                     # turning left
        cv2.rectangle(img, (mid_x - steer_fill, row_y),
                      (mid_x, row_y + bar_h),
                      (255, 160, 0), -1)
    steer_sign = "R" if steer >= 0 else "L"
    cv2.putText(img, f"{steer_sign}{abs(steer):.2f}",
                (bar_x + bar_max_w + 6, row_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 160, 0), 1)

    return img


def main():
    print(f"Starting Digital Twin receiver on {HOST}:{PORT} ...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print("Waiting for connection from InterFuser agent...")
    conn, addr = server.accept()
    print(f"✓ Connected by {addr}")

    try:
        while True:
            # 1) Read 4-byte length header
            header = recv_exact(conn, 4)
            if not header:
                print("Connection closed by client.")
                break
            msg_len = int.from_bytes(header, byteorder="big")

            # 2) Read full payload
            payload = recv_exact(conn, msg_len)
            if not payload:
                print("Connection closed while receiving payload.")
                break

            # 3) Unpickle state_data dict
            try:
                state_data = pickle.loads(payload)
            except Exception as e:
                print(f"Error unpickling data: {e}")
                continue

            images = state_data.get("images", {})
            bev_bytes = images.get("bev_sem_with_boxes", None)

            if bev_bytes is None:
                print("No bev_sem_with_boxes in this packet, skipping...")
                continue

            bev_img = jpeg_bytes_to_bgr(bev_bytes)
            if bev_img is None:
                continue

            # 4) Overlay: frame index and vehicle speed
            step = state_data.get("step", -1)
            veh = state_data.get("vehicle_state", {})
            bev_det = state_data.get("bev_detections", {})
            ctrl = state_data.get("control", {})

            speed_text = f"Frame {step} | Speed: {veh.get('velocity', 0):.1f} m/s"
            cv2.putText(bev_img, speed_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 5) Overlay: detection counts
            det_text = (f"V:{bev_det.get('vehicles', 0)} "
                        f"P:{bev_det.get('pedestrians', 0)} "
                        f"TL:{bev_det.get('traffic_lights', 0)} "
                        f"S:{bev_det.get('traffic_signs', 0)}")
            cv2.putText(bev_img, det_text, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            # 6) Driving command panel (throttle / brake / steer bars)
            if ctrl:
                throttle = float(ctrl.get("throttle", 0.0))
                steer    = float(ctrl.get("steer",    0.0))
                brake    = float(ctrl.get("brake",    0.0))
                bev_img  = draw_control_panel(bev_img, throttle, steer, brake)

            # 7) Show BEV map
            cv2.imshow("BEV HD Semantic Map", bev_img)

            # Press 'q' to quit, 's' to save screenshot
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"bev_frame_{step:06d}.png"
                cv2.imwrite(filename, bev_img)
                print(f"Saved: {filename}")

    finally:
        conn.close()
        server.close()
        cv2.destroyAllWindows()
        print("Receiver shut down.")


if __name__ == "__main__":
    main()
