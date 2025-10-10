import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import extrafuntions
from datetime import datetime
from pathlib import Path
import os
import time
import socket

from extrafuntions import camera_to_robot

#config ai model
MODEL_PATH = r"C:\Users\aashi\PycharmProjects\PythonProject\runs\detect\train08_10_25\weights\best.pt"
model = YOLO(MODEL_PATH)
SAVE_DIR = r"C:\AI-Hoeken_towels\review" # only for saving photos to look at it for review
IMG_SIZE = (1280, 736)
DISPLAY_SCALE = 0.7


Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# We only use these classes for the ai to detect
TARGET_CLASSES = ["towel", "corner"]

# Socket instellingen
#192.168.10.167 landuwasco
#192.168.137.10 robot
HOST_IP = "192.168.10.167"   # laptop
PORT = 5000

# Device kiezen
#if torch.cuda.is_available():
#    device = 0
#    use_half = True
#    print(" AI used on GPU")
#else:
device = "cpu"
use_half = False
print(" AI used on CPU")



#Realsense camera setup
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
pipe.start(cfg)
time.sleep(1)

#Function to check if there are detections of classes(towels or corners)
def detect_objects(image):
    #take frame
    results = model(image, imgsz=IMG_SIZE, device=device, half=use_half, stream=False)
    if not results or len(results) == 0:
        return [], image

    det = results[0]
    boxes = det.boxes
    annotated = image.copy()
    detections = []
    #Check all the boxes what the ai detected
    for i in range(len(boxes.xyxy)):
        cls_id = int(boxes.cls[i])
        cls_name = model.names.get(cls_id, str(cls_id)).lower()
        conf = float(boxes.conf[i])
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        #check if the boxes detected are in the allowed classes we made earlier
        if cls_name not in [t.lower() for t in TARGET_CLASSES]:
            continue

        detections.append((cls_name, conf, (x1, y1, x2, y2)))

        color = (0, 255, 0) if "towel" in cls_name else (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    return detections, annotated

#function to detect corner coords and angle
def detect_corner_angle(image, box):
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = vx.item(), vy.item()

    angle_deg = float(np.degrees(np.arctan2(vy, vx)))
    return angle_deg


#Socket server variables and starting it
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST_IP, PORT))
server_socket.listen(1)
print(f"Server started on ip and port: {HOST_IP}:{PORT}")
print("Wait for Doosan connection")

try:
    while True:
        #Wait for new connection
        conn, addr = server_socket.accept()
        print(f"Doosan connected, ip: {addr}")
        #variables for the loop logic
        check_value = False
        towel_found = False
        corner_found = False
        corner_data = None
        #loop for towel and corner detection
        while True:
            try:
                #Get the received data from the doosan
                data = conn.recv(1024).decode().strip()
                if not data:
                    print("Robot has lost connection.")
                    break

                print(f" Received data from doosan: {data}")

                # If the following string is in the data what the doosan said start checking for towel and corners
                if "CheckForCorner=1" in data:
                    check_value = True
                    towel_found = False
                    corner_found = False
                    corner_data = None
                    print("Start detection process")

                if check_value:
                    frames = pipe.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    #check if there are any objects detected by the ai
                    color_image = np.asanyarray(color_frame.get_data())
                    detections, annotated = detect_objects(color_image)

                    towel_boxes = [d for d in detections if "towel" in d[0]]
                    corner_boxes = [d for d in detections if "corner" in d[0]]
                    #if there is a towel go further else go back to detecting
                    if towel_boxes:
                        towel_found = True
                        cv2.putText(annotated, "Towel found", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    else:
                        cv2.putText(annotated, "No towel yet", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    #if towel is found and there are corners go further
                    if towel_found and corner_boxes:
                        best_corner = corner_boxes[0][2]
                        angle = detect_corner_angle(color_image, best_corner)
                        if angle is not None:
                            x1, y1, x2, y2 = best_corner
                            x_center = int((x1 + x2) / 2)
                            y_center = int((y1 + y2) / 2)
                            corner_found = True
                            corner_data = (x_center, y_center, angle)
                            cv2.circle(annotated, (x_center, y_center), 6, (255, 0, 0), -1)
                            cv2.putText(annotated, f"{angle:.1f}°", (x_center + 10, y_center),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    #camera output
                    disp = annotated.copy()
                    if DISPLAY_SCALE != 1.0:
                        w = int(disp.shape[1] * DISPLAY_SCALE)
                        h = int(disp.shape[0] * DISPLAY_SCALE)
                        disp = cv2.resize(disp, (w, h))
                    cv2.imshow("YOLO RealSense - Detectie", disp)


                    if towel_found:
                        print("TOwel found")
                        break

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        check_value = False
                        break

            except ConnectionResetError:
                print("Robot disconnected unexpectedly")
                break

        #Send the message back to the robot
        if towel_found:
            if corner_found and corner_data:
                x, y, angle = corner_data
                msg = (
                    f"TowelVisible=1;FirstCornerVisible=1;"
                    f"XfirstCorner={x:.3f};YfirstCorner={y:.3f};AngleFirstCorner={angle:.3f};"
                )
            else:
                msg = "TowelVisible=1;FirstCornerVisible=0;"
        else:
            msg = "TowelVisible=0;FirstCornerVisible=0;"

        conn.sendall(msg.encode())
        print(f"[INFO] Verstuurd aan robot: {msg}")

        # Save photo for the review afterwards
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
