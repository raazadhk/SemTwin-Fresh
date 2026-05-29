import socket
import os

# 🧠 Mapping of folder to Jetson IP
FOLDER_TO_JETSON = {
    "RGB": "192.168.0.122",
    "RGB_Left": "192.168.0.204",
    "RGB_Right": "192.168.0.166"
}

# 📍 Base directory containing Grad-CAM overlay results
BASE_DIR = "/home/carla1000/InterFuser/leaderboard/team_code/gradcam8_results"
PORT = 5001  # Port all Jetsons are listening on

# 📤 Send file to Jetson
def send_file(ip, port, file_path, relative_name):
    try:
        with socket.socket() as s:
            s.connect((ip, port))
            print(f"[CONNECTED] to {ip}:{port}")

            # Send file name length and name
            filename_bytes = relative_name.encode()
            s.send(len(filename_bytes).to_bytes(4, 'big'))
            s.send(filename_bytes)

            # Send file size
            filesize = os.path.getsize(file_path)
            s.send(filesize.to_bytes(8, 'big'))

            # Send file contents (Python 3.7–compatible version)
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    s.sendall(chunk)

            print(f"[SENT] {relative_name} to {ip}")

    except Exception as e:
        print(f"[ERROR] Failed to send {relative_name} to {ip}: {e}")

# 🔁 Go through each folder and send files
def main():
    print(f"[INFO] Looking in base directory: {BASE_DIR}")
    for folder, ip in FOLDER_TO_JETSON.items():
        input_dir = os.path.join(BASE_DIR, folder)
        print(f"[INFO] Checking folder: {input_dir} → Jetson IP: {ip}")

        if not os.path.exists(input_dir):
            print(f"[WARNING] Folder not found: {input_dir}")
            continue

        files = sorted(os.listdir(input_dir))
        if not files:
            print(f"[WARNING] No files found in {input_dir}")
            continue

        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            fpath = os.path.join(input_dir, fname)
            rel_name = f"{folder}/{fname}"
            print(f"[INFO] Sending file: {rel_name}")
            send_file(ip, PORT, fpath, rel_name)

if __name__ == "__main__":
    main()

