import socket
import pickle
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


HOST = "0.0.0.0"
PORT = 9999


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
    except Exception as exc:
        print(f"Error decoding image: {exc}")
        return None


def main():
    print(f"Starting Digital Twin receiver on {HOST}:{PORT} ...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print("Waiting for connection from InterFuser agent...")

    conn, addr = server.accept()
    print(f"Connected by {addr}")

    try:
        while True:
            header = recv_exact(conn, 4)
            if not header:
                print("Connection closed by client.")
                break

            msg_len = int.from_bytes(header, byteorder="big")
            payload = recv_exact(conn, msg_len)
            if not payload:
                print("Connection closed while receiving payload.")
                break

            try:
                state_data = pickle.loads(payload)
            except Exception as exc:
                print(f"Error unpickling data: {exc}")
                continue

            images = state_data.get("images", {})
            bev_bytes = images.get("bev_sem_with_boxes")
            if bev_bytes is None:
                print("No bev_sem_with_boxes in this packet, skipping...")
                continue

            bev_img = jpeg_bytes_to_bgr(bev_bytes)
            if bev_img is None:
                continue

            step = state_data.get("step", -1)
            veh = state_data.get("vehicle_state", {})
            bev_det = state_data.get("bev_detections", {})

            text = f"Frame {step} | Speed: {veh.get('velocity', 0):.1f} m/s"
            cv2.putText(
                bev_img,
                text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            det_text = (
                f"V:{bev_det.get('vehicles', 0)} "
                f"P:{bev_det.get('pedestrians', 0)} "
                f"TL:{bev_det.get('traffic_lights', 0)} "
                f"S:{bev_det.get('traffic_signs', 0)}"
            )
            cv2.putText(
                bev_img,
                det_text,
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )

            cv2.imshow("BEV HD Semantic Map", bev_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
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
