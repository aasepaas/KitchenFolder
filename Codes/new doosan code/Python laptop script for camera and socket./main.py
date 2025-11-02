import pyrealsense2 as rs
import numpy as np
import cv2
import time

# ---------- CONFIG ----------
DISPLAY_SCALE = 0.7  # display scale for showing the image
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FPS = 15
MARGIN = 50  # how far in from each corner the initial points will be

# ---------- RealSense setup ----------
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
pipe.start(cfg)
time.sleep(1)


# ---------- Variabelen ----------
# seed default points (corners but inset by MARGIN and the center)
def clamp_point(x, y):
    x = max(0, min(x, FRAME_WIDTH - 1))
    y = max(0, min(y, FRAME_HEIGHT - 1))
    return x, y


clicked_points = [
    clamp_point(MARGIN, MARGIN),  # top-left (inset)
    clamp_point(FRAME_WIDTH - MARGIN, MARGIN),  # top-right (inset)
    clamp_point(MARGIN, FRAME_HEIGHT - MARGIN),  # bottom-left (inset)
    clamp_point(FRAME_WIDTH - MARGIN, FRAME_HEIGHT - MARGIN),  # bottom-right (inset)
    clamp_point(FRAME_WIDTH // 2, FRAME_HEIGHT // 2)  # center
]

last_frame = None  # om beeldgrootte te kennen in callback


# ---------- Mouse callback ----------
def mouse_callback(event, x, y, flags, param):
    global clicked_points, last_frame
    if event == cv2.EVENT_LBUTTONDOWN and last_frame is not None:
        # Bereken klikpositie terug naar originele resolutie.
        # We displayed the image scaled by DISPLAY_SCALE, so convert back:
        if DISPLAY_SCALE != 0:
            inv_scale = 1.0 / DISPLAY_SCALE
        else:
            inv_scale = 1.0
        orig_x = int(x * inv_scale)
        orig_y = int(y * inv_scale)

        # clamp to image bounds
        orig_x = max(0, min(orig_x, FRAME_WIDTH - 1))
        orig_y = max(0, min(orig_y, FRAME_HEIGHT - 1))

        print(f"🖱️ Klik in window: ({x}, {y})  →  in origineel: X={orig_x}, Y={orig_y}")
        clicked_points.append((orig_x, orig_y))


# ---------- Window setup ----------
cv2.namedWindow("YOLO RealSense - Detectie", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("YOLO RealSense - Detectie", mouse_callback)

print("➡️ Klik met de linkermuisknop om X,Y te printen — druk op Q om te stoppen.")

# ---------- Main loop ----------
try:
    while True:
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        last_frame = color_image.copy()

        # teken klikpunten (in originele resolutie)
        for (x, y) in clicked_points:
            # draw circle and label (these coordinates are in original resolution)
            cv2.circle(color_image, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(color_image, f"({x},{y})", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # schaal voor weergave
        disp = color_image.copy()
        if DISPLAY_SCALE != 1.0:
            w = int(disp.shape[1] * DISPLAY_SCALE)
            h = int(disp.shape[0] * DISPLAY_SCALE)
            disp = cv2.resize(disp, (w, h))

        cv2.imshow("YOLO RealSense - Detectie", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()
    print("✅ Camera afgesloten")
