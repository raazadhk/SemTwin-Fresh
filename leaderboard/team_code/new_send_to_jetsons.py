#!/usr/bin/env python3
import socket
import os

# ✅ Mapping: camera folder → (Jetson IP, Jetson receiving port)
FOLDER_TO_JETSON = {
    "RGB": ("192.168.0.122", 5002),        # Jetson for RGB camera
    "RGB_Left": ("192.168.0.204", 5001),   # Jetson for RGB_Left camera
    "RGB_Right": ("192.168.0.166", 5003)   # Jetson for RGB_Right camera
}

# 📍 Base directory containing Grad-CAM overlay results
BASE_DIR = "/home/carla1000/InterFuser/leaderboard/team_code/gradcam8_results"

# 📤 Send one file to a Jetson
def send_file(ip, port, file_path, relative_name):
    try:
        with socket.socket() as s:
            s.connect((ip, port))
            print(f"[CONNECTED] to {ip}:{port}")

            # 🔹 Send file name length and name
            filename_bytes = relative_name.encode()
            s.send(len(filename_bytes).to_bytes(4, 'big'))
            s.send(filename_bytes)

            # 🔹 Send file size
            filesize = os.path.getsize(file_path)
            s.send(filesize.to_bytes(8, 'big'))

            # 🔹 Send file contents
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    s.sendall(chunk)

            print(f"[SENT] {relative_name} to {ip}:{port}")

    except Exception as e:
        print(f"[ERROR] Failed to send {relative_name} to {ip}:{port}: {e}")

# 🔁 Iterate over folders and send files to each Jetson
def main():
    print(f"[INFO] Scanning base directory: {BASE_DIR}")
    for folder, (ip, port) in FOLDER_TO_JETSON.items():
        input_dir = os.path.join(BASE_DIR, folder)
        print(f"[INFO] Checking folder: {input_dir} → Jetson IP: {ip}, Port: {port}")

        if not os.path.exists(input_dir):
            print(f"[WARNING] Folder not found: {input_dir}")
            continue

        files = sorted(os.listdir(input_dir))
        if not files:
            print(f"[WARNING] No image files found in {input_dir}")
            continue

        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            file_path = os.path.join(input_dir, fname)
            relative_name = f"{folder}/{fname}"
            print(f"[INFO] Sending file: {relative_name}")
            send_file(ip, port, file_path, relative_name)

if __name__ == "__main__":
    main()

