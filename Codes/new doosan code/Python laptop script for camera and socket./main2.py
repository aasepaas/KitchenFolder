import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from extraZ import camera_to_robotZCoords
from functieAngleVerandering import cameraAngle_to_doosan
import time
import socket

# === CONFIGURATIE ===
MODEL_PATH = r"C:\Users\aashi\PycharmProjects\PythonProject\runs\detect\train18\weights\best.pt"
TARGET_CLASSES = ["towel", "corner"]
HOST_IP = "192.168.137.10"
PORT = 5000
IMG_SIZE = (1280, 736)
device = "cpu"
use_half = False
maxAngleRetries = 6

# === ROI-INSTELLING ===
ROI_COORDS = (521, 850, 1662, 60)

# === MODEL LADEN ===
model = YOLO(MODEL_PATH)
print("AI model geladen (CPU)")

# === REALSENSE SETUP ===
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
pipe.start(cfg)
time.sleep(1)


def normalize_roi(coords, img_shape):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = map(int, coords)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
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
    return cv2.bitwise_and(image, image, mask=mask)


def detect_objects(image):
    results = model(image, imgsz=IMG_SIZE, device=device, half=use_half, stream=False)
    if not results or len(results) == 0:
        return []

    det = results[0]
    boxes = det.boxes
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

    return detections


def detect_corner_angle(roi_image, box=None):
    if box is not None:
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
        return None

    cnt = max(contours, key=cv2.contourArea)
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

    raw_deg = np.degrees(avg_ang)
    base_angle = ((raw_deg + 180) % 360) - 180

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

    alt_angle = ((base_angle + 180) + 180) % 360 - 180
    candidates = [base_angle, alt_angle]

    vec_x = corner_x - cx
    vec_y = cy - corner_y
    if vec_x == 0 and vec_y == 0:
        chosen_angle = base_angle
    else:
        corner_dir_deg = ((np.degrees(np.arctan2(vec_y, vec_x)) + 180) % 360) - 180
        diffs = [abs(((c - corner_dir_deg + 180) % 360) - 180) for c in candidates]
        chosen_angle = candidates[int(np.argmin(diffs))]

    return chosen_angle


# === SOCKET SERVER START ===
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST_IP, PORT))
server_socket.listen(1)
print(f"Server gestart op {HOST_IP}:{PORT}")

try:
    while True:
        conn, addr = server_socket.accept()
        print(f"Verbonden met robot: {addr}")

        check_value = False
        towel_found = False
        corner_found = False
        corner_data = None
        towel_frame_counter = 0

        while True:
            try:
                conn.setblocking(False)
                try:
                    data = conn.recv(1024).decode().strip()
                    if not data:
                        break
                    if "CheckForCorner=1" in data:
                        check_value = True
                        towel_found = False
                        corner_found = False
                        corner_data = None
                        towel_frame_counter = 0
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
                    x1, y1, x2, y2 = normalize_roi(ROI_COORDS, color_image.shape)
                    masked_image = apply_roi_mask(color_image, (x1, y1, x2, y2))
                    detections = detect_objects(masked_image)

                    towel_boxes = [d for d in detections if "towel" in d[0]]
                    corner_boxes = [d for d in detections if "corner" in d[0]]

                    if towel_boxes:
                        towel_found = True
                    else:
                        towel_found = False

                    if towel_found and not corner_boxes:
                        towel_frame_counter += 1
                    else:
                        towel_frame_counter = 0

                    if towel_frame_counter >= maxAngleRetries:
                        check_value = False
                        break

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
                                continue

                            color_image = np.asanyarray(color_frame.get_data())
                            masked_image = apply_roi_mask(color_image, normalize_roi(ROI_COORDS, color_image.shape))
                            detections = detect_objects(masked_image)
                            new_corner_boxes = [d for d in detections if "corner" in d[0]]

                            candidates = []
                            if new_corner_boxes:
                                for cls_name, conf, bbox in sorted(new_corner_boxes, key=lambda d: d[1], reverse=True):
                                    ang = detect_corner_angle(masked_image, bbox)
                                    if ang is not None:
                                        candidates.append((conf, bbox, ang))
                            else:
                                ang = detect_corner_angle(masked_image, fallback_bbox)
                                if ang is not None:
                                    candidates.append((fallback_conf, fallback_bbox, ang))

                            if candidates:
                                sel_conf, sel_bbox, sel_angle = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
                                x1_box, y1_box, x2_box, y2_box = sel_bbox
                                x_center = int((x1_box + x2_box) / 2)
                                y_center = int((y1_box + y2_box) / 2)
                                corner_found = True
                                corner_data = (x_center, y_center, sel_angle)
                                found = True
                                break

                        if not found:
                            check_value = False
                            break

                    if towel_found and corner_found:
                        check_value = False
                        break

            except ConnectionResetError:
                break

        if towel_found:
            if corner_found and corner_data:
                x, y, angle = corner_data
                xVerwerkt, yVerwerkt, zVerwerkt = camera_to_robotZCoords(x, y)
                _, _, zVerwerkt2 = camera_to_robotZCoords(x, y)
                Cverwerkt = cameraAngle_to_doosan(angle)
                msg = (
                    f"TowelVisible=1;FirstCornerVisible=1;"
                    f"XfirstCorner={xVerwerkt:.3f};YfirstCorner={yVerwerkt:.3f};"
                    f"Z={zVerwerkt2:.3f};C={Cverwerkt:.3f};"
                )
            else:
                msg = "TowelVisible=1;FirstCornerVisible=0;"
        else:
            msg = "TowelVisible=0;FirstCornerVisible=0;"

        conn.sendall(msg.encode())
        conn.close()
        print("Resultaat verstuurd en verbinding gesloten\n")

finally:
    pipe.stop()
    server_socket.close()
    print("Camera en socket afgesloten.")
