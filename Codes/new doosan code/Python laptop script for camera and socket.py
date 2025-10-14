import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from extrafuntions import camera_to_robot
from datetime import datetime
from pathlib import Path
import os
import time
import socket

# ---------------------------
# Config AI model
# ---------------------------
MODEL_PATH = r"C:\Users\aashi\PycharmProjects\PythonProject\runs\detect\train16\weights\best.pt"
model = YOLO(MODEL_PATH)
SAVE_DIR = r"C:\AI-Hoeken_towels\review"  # only for saving photos to look at it for review
IMG_SIZE = (1280, 736)
DISPLAY_SCALE = 0.7
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["towel", "corner"]

# ---------------------------
# Socket instellingen
# ---------------------------
HOST_IP = "192.168.10.167"  # laptop
PORT = 5000

# ---------------------------
# Device setup
# ---------------------------
device = "cpu"
use_half = False
print(" AI used on CPU")

# ---------------------------
# Realsense camera setup
# ---------------------------
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
pipe.start(cfg)
time.sleep(1)

# ---------------------------
# Functies
# ---------------------------
def detect_objects(image):
    results = model(image, imgsz=IMG_SIZE, device=device, half=use_half, stream=False)
    if not results or len(results) == 0:
        return [], image

    det = results[0]
    boxes = det.boxes
    annotated = image.copy()
    detections = []

    for i in range(len(boxes.xyxy)):
        cls_id = int(boxes.cls[i])
        cls_name = model.names.get(cls_id, str(cls_id)).lower()
        conf = float(boxes.conf[i])
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        if cls_name not in [t.lower() for t in TARGET_CLASSES]:
            continue

        detections.append((cls_name, conf, (x1, y1, x2, y2)))
        color = (0, 255, 0) if "towel" in cls_name else (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    return detections, annotated

# ---------------------------
# Corner angle detectie met visuele lijnen
# ---------------------------
def detect_corner_angle(image, box, annotated):
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    data = cnt.reshape(-1, 2).astype(np.float32)

    mean, eigenvectors = cv2.PCACompute(data, mean=np.array([]))
    projections = (data - mean) @ eigenvectors[0]
    median_val = np.median(projections)
    cluster1 = data[projections < median_val]
    cluster2 = data[projections >= median_val]

    lines = []
    for cluster in [cluster1, cluster2]:
        if len(cluster) < 2:
            continue
        vx, vy, x0, y0 = cv2.fitLine(cluster, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
        lines.append((vx, vy, x0, y0))

    points = []
    angle_deg = None

    if len(lines) == 2:
        for (vx, vy, x0, y0) in lines:
            x1_line = int(x0 - vx * 1000)
            y1_line = int(y0 - vy * 1000)
            x2_line = int(x0 + vx * 1000)
            y2_line = int(y0 + vy * 1000)
            cv2.line(roi, (x1_line, y1_line), (x2_line, y2_line), (255, 0, 0), 2)
            points.append((x1_line, y1_line, x2_line, y2_line))

        def line_intersection(p1, p2):
            x1, y1, x2, y2 = p1
            x3, y3, x4, y4 = p2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return int((x1 + x2 + x3 + x4) / 4), int((y1 + y2 + y3 + y4) / 4)
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return int(px), int(py)

        ix, iy = line_intersection(points[0], points[1])

        angle1 = np.arctan2(lines[0][1], lines[0][0])
        angle2 = np.arctan2(lines[1][1], lines[1][0])
        avg_angle = (angle1 + angle2) / 2
        angle_deg = float(np.degrees(avg_angle))

        L = 150
        dx = int(L * np.cos(avg_angle))
        dy = int(L * np.sin(avg_angle))
        cv2.line(roi, (ix - dx, iy - dy), (ix + dx, iy + dy), (0, 255, 0), 2)
        cv2.circle(roi, (ix, iy), 5, (0, 255, 0), -1)

        cv2.putText(annotated, f"{angle_deg:.1f}°", (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    annotated[y1:y2, x1:x2] = roi
    return angle_deg

# ---------------------------
# Socket server starten
# ---------------------------
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST_IP, PORT))
server_socket.listen(1)
print(f"Server started on ip and port: {HOST_IP}:{PORT}")
print("Wait for Doosan connection")

try:
    while True:
        conn, addr = server_socket.accept()
        print(f"Doosan connected, ip: {addr}")

        check_value = False
        towel_found = False
        corner_found = False
        corner_data = None

        while True:
            try:
                # Use non-blocking recv to check for incoming messages
                conn.setblocking(False)
                try:
                    data = conn.recv(1024).decode().strip()
                    if not data:
                        print("Robot has lost connection.")
                        conn.setblocking(True)
                        break
                    print(f" Received data from doosan: {data}")
                    if "CheckForCorner=1" in data:
                        check_value = True
                        towel_found = False
                        corner_found = False
                        corner_data = None
                        print("Start detection process")
                except BlockingIOError:
                    # No data received, continue with detection loop
                    pass
                finally:
                    conn.setblocking(True)

                # CONTINUOUS DETECTION LOOP
                if check_value:
                    frames = pipe.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    detections, annotated = detect_objects(color_image)
                    towel_boxes = [d for d in detections if "towel" in d[0]]
                    corner_boxes = [d for d in detections if "corner" in d[0]]

                    if towel_boxes:
                        towel_found = True
                        cv2.putText(annotated, "Towel found", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    else:
                        cv2.putText(annotated, "No towel yet", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    if towel_found and corner_boxes and not corner_found:
                        best_corner = corner_boxes[0][2]
                        angle = detect_corner_angle(color_image, best_corner, annotated)
                        if angle is not None:
                            x1, y1, x2, y2 = best_corner
                            x_center = int((x1 + x2) / 2)
                            y_center = int((y1 + y2) / 2)
                            corner_found = True
                            corner_data = (x_center, y_center, angle)
                            cv2.circle(annotated, (x_center, y_center), 6, (255, 0, 0), -1)
                            cv2.putText(annotated, f"{angle:.1f}°", (x_center + 10, y_center),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    disp = annotated.copy()
                    if DISPLAY_SCALE != 1.0:
                        w = int(disp.shape[1] * DISPLAY_SCALE)
                        h = int(disp.shape[0] * DISPLAY_SCALE)
                        disp = cv2.resize(disp, (w, h))

                    cv2.imshow("YOLO RealSense - Detectie", disp)

                    # Exit detection loop only when both found
                    if towel_found and corner_found:
                        print("Towel and corner found")
                        check_value = False
                        break

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        check_value = False
                        break

            except ConnectionResetError:
                print("Robot disconnected unexpectedly")
                break

        # Verstuur resultaat naar robot
        if towel_found:
            if corner_found and corner_data:
                x, y, angle = corner_data
                xVerwerkt, yverwerkt = camera_to_robot(x, y)
                msg = (
                    f"TowelVisible=1;FirstCornerVisible=1;"
                    f"XfirstCorner={xVerwerkt:.3f};YfirstCorner={yverwerkt:.3f};AngleFirstCorner={angle:.3f};"
                )
            else:
                msg = "TowelVisible=1;FirstCornerVisible=0;"
        else:
            msg = "TowelVisible=0;FirstCornerVisible=0;"

        conn.sendall(msg.encode())
        print(f"Verstuurd to robot: {msg}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SAVE_DIR, f"result_{timestamp}.jpg")
        if 'disp' in locals():
            cv2.imwrite(save_path, disp)
            print(f"Resultaat opgeslagen als: {save_path}")
        else:
            print("Geen beeld om op te slaan (detectie niet uitgevoerd).")

        conn.close()
        print("Connection closed\n")

finally:
    print("Finalizing, closing camera, socket and all open windows")
    pipe.stop()
    server_socket.close()
    cv2.destroyAllWindows()
