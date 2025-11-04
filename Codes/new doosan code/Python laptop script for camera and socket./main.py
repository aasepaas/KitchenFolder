import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
#from extrafuntions import camera_to_robot
from extraZ import camera_to_robotZCoords
from functieAngleVerandering import cameraAngle_to_doosan
from datetime import datetime
from pathlib import Path
import os
import time
import socket

# Config AI model
MODEL_PATH = r"C:\Users\aashi\PycharmProjects\PythonProject\runs\detect\train18\weights\best.pt"
model = YOLO(MODEL_PATH)
SAVE_DIR = r"C:\AI-Hoeken_towels\review"
IMG_SIZE = (1280, 736)
DISPLAY_SCALE = 0.7
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["towel", "corner"]

# Socket variables
HOST_IP = "192.168.137.10"
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

# Max retries variables for angle calculations and corner check after towel is found
maxAngleRetries = 6

# ===== USER SET ROI (x1, y1, x2, y2) in ORIGINAL IMAGE COORDINATES =====
# Stel hier je ROI in (pixelwaarden voor full-res frame)
ROI_COORDS = (521, 850, 1662, 60)
# =====================================================================

def normalize_roi(coords, img_shape):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = map(int, coords)
    # swap indien nodig
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    # clip binnen beeld
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2

def apply_roi_mask(image, roi):
    x1, y1, x2, y2 = roi
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.rectangle(masked, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return masked

# Functions (ongeveer ongewijzigd)
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






def detect_corner_angle(roi_image, box=None, annotated=None, visualize=True):

    # Zorg dat we roi hebben als crop
    if box is not None and roi_image is not None and annotated is not None:
        x1, y1, x2, y2 = box
        roi = roi_image[y1:y2, x1:x2].copy()
    else:
        roi = roi_image.copy()

    if roi.size == 0:
        return None

    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if visualize:
            cv2.putText(roi, "No contours", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if annotated is not None and box is not None:
                annotated[y1:y2, x1:x2] = roi
        return None

    cnt = max(contours, key=cv2.contourArea)

    if visualize:
        cv2.drawContours(roi, [cnt], -1, (0, 255, 255), 2)

    pts = cnt.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 2:
        return None

    try:
        mean, eigenvectors = cv2.PCACompute(pts, mean=np.array([]))
    except Exception:
        return None
    projections = (pts - mean) @ eigenvectors[0]
    median_val = np.median(projections)
    cl1 = pts[projections < median_val]
    cl2 = pts[projections >= median_val]

    lines = []
    for cl in (cl1, cl2):
        if len(cl) < 2:
            continue
        vx, vy, x0, y0 = cv2.fitLine(cl, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx[0]), float(vy[0])
        x0, y0 = float(x0[0]), float(y0[0])
        lines.append((vx, vy, x0, y0))

    if len(lines) < 2:
        return None

    ang1 = np.arctan2(lines[0][1], lines[0][0])
    ang2 = np.arctan2(lines[1][1], lines[1][0])
    sx = np.cos(ang1) + np.cos(ang2)
    sy = np.sin(ang1) + np.sin(ang2)
    if abs(sx) < 1e-9 and abs(sy) < 1e-9:
        avg_ang = (ang1 + ang2) / 2.0
    else:
        avg_ang = np.arctan2(sy, sx)

    # raw angle in degrees (0 means pointing right, positive up)
    raw_deg = np.degrees(avg_ang)
    # normaliseer basiswaarde naar (-180,180]
    base_angle = ((raw_deg + 180) % 360) - 180

    # Bereken snijpunt van lijnen als hoekpunt
    def line_to_params(vx, vy, x0, y0):
        a = vy
        b = -vx
        c = vx * y0 - vy * x0
        return a, b, c

    a1, b1, c1 = line_to_params(*lines[0])
    a2, b2, c2 = line_to_params(*lines[1])
    det = a1 * b2 - a2 * b1
    cx, cy = w // 2, h // 2
    if abs(det) < 1e-6:
        corner_x, corner_y = cx, cy
    else:
        xi = (b1 * c2 - b2 * c1) / det
        yi = (c1 * a2 - c2 * a1) / det
        corner_x, corner_y = int(round(xi)), int(round(yi))

    # Om de 180°-ambiguïteit te fixen: maak twee kandidaten (base_angle en base_angle +/- 180)
    alt_angle = ((base_angle + 180) + 180) % 360 - 180  # base_angle +/- 180 normalized
    candidates = [base_angle, alt_angle]

    # Bereken richting van het hoekpunt t.o.v. het midden van de ROI
    vec_x = corner_x - cx
    vec_y = cy - corner_y  # y-omkering zodat positieve hoek omhoog blijft zoals raw_deg
    if vec_x == 0 and vec_y == 0:
        chosen_angle = base_angle
    else:
        corner_dir_deg = ((np.degrees(np.arctan2(vec_y, vec_x)) + 180) % 360) - 180
        # Kies kandidaat die het dichtst bij hoekrichting ligt (minimale absolute afwijking)
        diffs = [abs(((c - corner_dir_deg + 180) % 360) - 180) for c in candidates]
        chosen_angle = candidates[int(np.argmin(diffs))]

    angle_deg = chosen_angle  # deze waarde geven we terug

    # Visualisatie
    if visualize:
        cv2.line(roi, (0, cy), (w, cy), (0, 0, 255), 2)
        cv2.line(roi, (w//2, 0), (w//2, h), (0, 0, 0), 2)
        long_len = max(w, h)
        for (vx, vy, x0, y0) in lines:
            x0_i = int(round(x0))
            y0_i = int(round(y0))
            dx = int(round(long_len * vx))
            dy = int(round(long_len * vy))
            p1 = (x0_i - dx, y0_i - dy)
            p2 = (x0_i + dx, y0_i + dy)
            cv2.line(roi, p1, p2, (255, 0, 0), 2)

        cv2.circle(roi, (corner_x, corner_y), 6, (0, 255, 0), -1)
        cv2.circle(roi, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(roi, f"{angle_deg:+.1f}°", (corner_x + 10, corner_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    if visualize and annotated is not None and box is not None:
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

                if check_value:
                    frames = pipe.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())

                    # Normaliseer ROI t.o.v. dit frame
                    x1, y1, x2, y2 = normalize_roi(ROI_COORDS, color_image.shape)

                    # Apply ROI mask (alles buiten ROI zwart)
                    masked_image = apply_roi_mask(color_image, (x1, y1, x2, y2))

                    # detecties op masked_image (coördinaten refereren nog steeds aan full-image)
                    detections, annotated = detect_objects(masked_image)

                    # Zorg dat polygon/rect zichtbaar blijft op annotated (indien detect_objects over zwart tekent)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
                            x1, y1, x2, y2 = normalize_roi(ROI_COORDS, color_image.shape)
                            masked_image = apply_roi_mask(color_image, (x1, y1, x2, y2))
                            detections, annotated = detect_objects(masked_image)
                            current_annotated = annotated

                            new_corner_boxes = [d for d in detections if "corner" in d[0]]

                            candidates = []

                            if new_corner_boxes:
                                for cls_name, conf, bbox in sorted(new_corner_boxes, key=lambda d: d[1], reverse=True):
                                    ang = detect_corner_angle(masked_image, bbox, annotated)
                                    print(f"Attempt {attempt}: tried bbox conf={conf:.3f} -> angle={ang}")
                                    if ang is not None:
                                        candidates.append((conf, bbox, ang))
                            else:
                                ang = detect_corner_angle(masked_image, fallback_bbox, annotated)
                                print(f"Attempt {attempt}: no new corner detections, tried fallback conf={fallback_conf:.3f} -> angle={ang}")
                                if ang is not None:
                                    candidates.append((fallback_conf, fallback_bbox, ang))

                            if candidates:
                                candidates.sort(key=lambda x: x[0], reverse=True)
                                sel_conf, sel_bbox, sel_angle = candidates[0]

                                x1_box, y1_box, x2_box, y2_box = sel_bbox
                                x_center = int((x1_box + x2_box) / 2)
                                y_center = int((y1_box + y2_box) / 2)

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

                    cv2.imshow("YOLO RealSense - Detectie (ROI-only)", disp)

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
                xVerwerkt, yverwerkt, zVerwerkt = camera_to_robotZCoords(x, y)
                xoud, youd, zVerwerkt2 = camera_to_robotZCoords(x,y)
                Cverwerkt = cameraAngle_to_doosan(angle)
                print(f"Angle: {angle}, verwerkt: {Cverwerkt}")
                print(f"X en Y : {x}  {y}")
                msg = (
                    f"TowelVisible=1;FirstCornerVisible=1;"
                    f"XfirstCorner={xVerwerkt:.3f};YfirstCorner={yverwerkt:.3f};Z={zVerwerkt2:.3f};C={Cverwerkt:.3f};"
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
        save_path_roi_raw = os.path.join(SAVE_DIR, f"result_{timestamp}_roi_raw.jpg")
        save_path_roi_annot = os.path.join(SAVE_DIR, f"result_{timestamp}_roi_annotated.jpg")

        # Sla annotated full-frame op (eventueel geschaald voor overzicht)
        if current_annotated is not None:
            disp_to_save = current_annotated.copy()
            # Bewaar full-res ROI apart; deze saved below
            if DISPLAY_SCALE != 1.0:
                w = int(disp_to_save.shape[1] * DISPLAY_SCALE)
                h = int(disp_to_save.shape[0] * DISPLAY_SCALE)
                disp_to_save = cv2.resize(disp_to_save, (w, h))
            cv2.imwrite(save_path_annotated, disp_to_save)
            print(f"Annotated image opgeslagen als: {save_path_annotated}")
        else:
            print("Geen annotated beeld om op te slaan (detectie niet uitgevoerd).")

        # Probeer de laatste originele frame ook te bewaren en aparte ROI-crop
        try:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                raw_image = np.asanyarray(color_frame.get_data())
                cv2.imwrite(save_path_raw, raw_image)
                print(f"Raw (onnannotated) image opgeslagen als: {save_path_raw}")

                # crop ROI van raw_image (gebruik normalize zodat binnen bounds)
                x1c, y1c, x2c, y2c = normalize_roi(ROI_COORDS, raw_image.shape)
                if x2c > x1c and y2c > y1c:
                    roi_raw = raw_image[y1c:y2c, x1c:x2c].copy()
                    cv2.imwrite(save_path_roi_raw, roi_raw)
                    print(f"Raw ROI image opgeslagen als: {save_path_roi_raw}")
                else:
                    print("ROI ongeldig om raw crop op te slaan.")
            else:
                print("Geen raw frame beschikbaar om op te slaan.")
        except Exception as e:
            print(f"Kon raw afbeelding niet opslaan: {e}")

        # Sla ook annotated ROI (uit current_annotated full-res) op
        try:
            if current_annotated is not None:

                x1c, y1c, x2c, y2c = normalize_roi(ROI_COORDS, current_annotated.shape)
                if x2c > x1c and y2c > y1c:
                    roi_annot = current_annotated[y1c:y2c, x1c:x2c].copy()
                    cv2.imwrite(save_path_roi_annot, roi_annot)
                    print(f"Annotated ROI image opgeslagen als: {save_path_roi_annot}")
                else:
                    print("ROI ongeldig om annotated crop op te slaan.")
        except Exception as e:
            print(f"Kon annotated ROI niet opslaan: {e}")

        conn.close()
        print("Connection closed\n")

finally:
    print("Finalizing, closing camera, socket and all open windows")
    pipe.stop()
    server_socket.close()
    cv2.destroyAllWindows()
