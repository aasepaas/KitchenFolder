


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


# Config AI model
MODEL_PATH = r"C:\Users\aashi\PycharmProjects\PythonProject\runs\detect\train16\weights\best.pt"
model = YOLO(MODEL_PATH)
SAVE_DIR = r"C:\AI-Hoeken_towels\review"  # only for saving photos to look at it for review, nothing added to the general code
IMG_SIZE = (1280, 736)
DISPLAY_SCALE = 0.7
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["towel", "corner"]

# Socket variables
HOST_IP = "192.168.137.10"  # laptop
PORT = 5000


# Device setup
device = "cpu"
use_half = False
print(" AI used on CPU")

# Realsense camera setup
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
pipe.start(cfg)
time.sleep(1)

#Max retries variables for angle calculations and corner check after towel is found
maxAngleRetries = 10

# Functions
# function to check all the objects in the image
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


# Corner angle detectie met WHITE MASK CHECK + visuele lijnen + retry logica
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
        vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])
        lines.append((vx, vy, x0, y0))

    angle_deg = None
    if len(lines) == 2:
        def line_intersection(p1, p2):
            x1, y1, x2, y2 = p1
            x3, y3, x4, y4 = p2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return int((x1 + x2 + x3 + x4) / 4), int((y1 + y2 + y3 + y4) / 4)
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return int(px), int(py)

        pts = []
        for (vx, vy, x0, y0) in lines:
            x1l, y1l = int(x0 - vx), int(y0 - vy)
            x2l, y2l = int(x0 + vx), int(y0 + vy)
            pts.append((x1l, y1l, x2l, y2l))

        ix, iy = line_intersection(pts[0], pts[1])
        angle1 = np.arctan2(lines[0][1], lines[0][0])
        angle2 = np.arctan2(lines[1][1], lines[1][0])
        avg_angle = (angle1 + angle2) / 2
        angle_deg = float(np.degrees(avg_angle))

        if angle_deg < -45 or angle_deg > 45:
            return None

        # ===== NIEUWE WHITE MASK CHECK =====
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Strategie 1: HSV white mask (relaxed)
        lower_white_hsv = np.array([0, 0, 120])
        upper_white_hsv = np.array([180, 100, 255])
        white_mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

        # Strategie 2: Grayscale brightness mask
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white_mask_gray = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY)

        # Combineer beide masks
        white_mask = cv2.bitwise_or(white_mask_hsv, white_mask_gray)

        # Morfologische operaties
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # Sample punten langs de groene lijn
        L = 150
        samples = 40
        white_count = 0
        total_valid_samples = 0

        for i in range(-samples // 2, samples // 2):
            t = (i / samples) * L
            x = int(ix + t * np.cos(avg_angle))
            y = int(iy + t * np.sin(avg_angle))

            if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                total_valid_samples += 1
                if white_mask[y, x] > 0:
                    white_count += 1

        if total_valid_samples == 0:
            print(f" iGeen valide samples binnen ROI")
            return None

        white_ratio = white_count / total_valid_samples
        print(f"    White coverage: {white_ratio:.1%} ({white_count}/{total_valid_samples})")

        # Threshold check
        MIN_WHITE_RATIO = 0.10
        if white_ratio < MIN_WHITE_RATIO:
            print(f" Raaklijn gedetecteerd (slechts {white_ratio:.1%} op wit), verwerpen")
            return None

        print(f"Check geslaagd: {white_ratio:.1%} op wit")
        # ===================================

        # Teken lijnen op ROI
        dx = int(L * np.cos(avg_angle))
        dy = int(L * np.sin(avg_angle))

        # Teken blauwe edge lijnen
        for (vx, vy, x0, y0) in lines:
            x1l, y1l = int(x0 - vx), int(y0 - vy)
            x2l, y2l = int(x0 + vx), int(y0 + vy)
            cv2.line(roi, (x1l, y1l), (x2l, y2l), (255, 0, 0), 2)

        # Teken groene hoeklijn
        cv2.line(roi, (ix - dx, iy - dy), (ix + dx, iy + dy), (0, 255, 0), 3)
        cv2.circle(roi, (ix, iy), 5, (0, 255, 0), -1)

        annotated[y1:y2, x1:x2] = roi

    return angle_deg


# start Socket server
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
        current_annotated = None
        towel_frame_counter = 0

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
                        towel_frame_counter = 0
                        print("Start detection process")
                except BlockingIOError:
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
                    current_annotated = annotated

                    towel_boxes = [d for d in detections if "towel" in d[0]]
                    corner_boxes = [d for d in detections if "corner" in d[0]]

                    if towel_boxes:
                        towel_found = True
                        cv2.putText(annotated, "Towel found", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    else:
                        cv2.putText(annotated, "No towel yet", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    if towel_found and not corner_boxes:
                        towel_frame_counter += 1
                        print(f"Towel without corner for {towel_frame_counter} frames")
                    else:
                        towel_frame_counter = 0

                    if towel_frame_counter >= maxAngleRetries:
                        print(f"Towel gedurende {towel_frame_counter} frames zonder hoek — doorgaan en resultaat sturen.")
                        check_value = False
                        break

                    # ANGLE DETECTION WITH RETRY LOGIC
                    if towel_found and corner_boxes and not corner_found:
                        corner_boxes_sorted = sorted(corner_boxes, key=lambda d: d[1], reverse=True)
                        fallback_bbox = corner_boxes_sorted[0][2]
                        fallback_conf = corner_boxes_sorted[0][1]

                        attempt = 0
                        found = False

                        while attempt < maxAngleRetries and not found:
                            attempt += 1
                            frames = pipe.wait_for_frames()
                            color_frame = frames.get_color_frame()
                            if not color_frame:
                                print(f"No color frame on attempt {attempt}, continuing")
                                continue

                            color_image = np.asanyarray(color_frame.get_data())
                            detections, annotated = detect_objects(color_image)
                            current_annotated = annotated

                            new_corner_boxes = [d for d in detections if "corner" in d[0]]

                            candidates = []

                            if new_corner_boxes:
                                for cls_name, conf, bbox in sorted(new_corner_boxes, key=lambda d: d[1], reverse=True):
                                    ang = detect_corner_angle(color_image, bbox, annotated)
                                    print(f"Attempt {attempt}: tried bbox conf={conf:.3f} -> angle={ang}")
                                    if ang is not None:
                                        candidates.append((conf, bbox, ang))
                            else:
                                ang = detect_corner_angle(color_image, fallback_bbox, annotated)
                                print(f"Attempt {attempt}: no new corner detections, tried fallback conf={fallback_conf:.3f} -> angle={ang}")
                                if ang is not None:
                                    candidates.append((fallback_conf, fallback_bbox, ang))

                            if candidates:
                                candidates.sort(key=lambda x: x[0], reverse=True)
                                sel_conf, sel_bbox, sel_angle = candidates[0]

                                x1, y1, x2, y2 = sel_bbox
                                x_center = int((x1 + x2) / 2)
                                y_center = int((y1 + y2) / 2)

                                corner_found = True
                                corner_data = (x_center, y_center, sel_angle)

                                cv2.circle(annotated, (x_center, y_center), 6, (255, 0, 0), -1)
                                cv2.putText(annotated, f"{sel_angle:.1f}°", (x_center + 10, y_center),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                current_annotated = annotated.copy()
                                found = True
                                print(f"Selected corner (conf={sel_conf:.3f}) with angle={sel_angle:.2f} on attempt {attempt}")
                                break

                            print(f"Attempt {attempt} finished — no valid angle found yet.")

                        if not found:
                            print(f"Angle not found after {attempt} attempts — proceeding to send result")
                            check_value = False
                            break

                    disp = annotated.copy()
                    if DISPLAY_SCALE != 1.0:
                        w = int(disp.shape[1] * DISPLAY_SCALE)
                        h = int(disp.shape[0] * DISPLAY_SCALE)
                        disp = cv2.resize(disp, (w, h))

                    cv2.imshow("YOLO RealSense - Detectie", disp)

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
                Cverwerkt = 0.958 * angle - 83.68
                print(f"Angle: {angle}, verwerkt: {Cverwerkt}")
                msg = (
                    f"TowelVisible=1;FirstCornerVisible=1;"
                    f"XfirstCorner={xVerwerkt:.3f};YfirstCorner={yverwerkt:.3f};C={Cverwerkt:.3f};"
                )
            else:
                msg = "TowelVisible=1;FirstCornerVisible=0;"
        else:
            msg = "TowelVisible=0;FirstCornerVisible=0;"

        conn.sendall(msg.encode())
        print(f"Verstuurd to robot: {msg}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Bestandsnamen voorbereiden
        save_path_annotated = os.path.join(SAVE_DIR, f"result_{timestamp}_annotated.jpg")
        save_path_raw = os.path.join(SAVE_DIR, f"result_{timestamp}_raw.jpg")

        if current_annotated is not None:
            # Annotated beeld opslaan
            disp_to_save = current_annotated.copy()
            if DISPLAY_SCALE != 1.0:
                w = int(disp_to_save.shape[1] * DISPLAY_SCALE)
                h = int(disp_to_save.shape[0] * DISPLAY_SCALE)
                disp_to_save = cv2.resize(disp_to_save, (w, h))
            cv2.imwrite(save_path_annotated, disp_to_save)
            print(f"🖼 Annotated image opgeslagen als: {save_path_annotated}")
        else:
            print("⚠️ Geen annotated beeld om op te slaan (detectie niet uitgevoerd).")

        # Probeer de laatste originele frame ook te bewaren
        try:
            # Probeer het laatste frame van de RealSense pipeline op te halen
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                raw_image = np.asanyarray(color_frame.get_data())
                cv2.imwrite(save_path_raw, raw_image)
                print(f"📸 Raw (onnannotated) image opgeslagen als: {save_path_raw}")
            else:
                print("⚠️ Geen raw frame beschikbaar om op te slaan.")
        except Exception as e:
            print(f"⚠️ Kon raw afbeelding niet opslaan: {e}")

        conn.close()
        print("Connection closed\n")

finally:
    print("Finalizing, closing camera, socket and all open windows")
    pipe.stop()
    server_socket.close()
    cv2.destroyAllWindows()
