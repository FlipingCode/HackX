import os
import cv2
import time
import numpy as np
from inference import get_model
from ultralytics import YOLO

os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"

# --- SETUP MODELS ---
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"
MODEL_ID = "drawer-alerts/1"
drawer_model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
pose_model = YOLO('yolov8n-pose.pt')

# --- ZONES & THRESHOLDS ---
CASHIER_ZONE = np.array(
    [[600, 100], [1200, 100], [1200, 700], [600, 700]], np.int32)
CUSTOMER_ZONE = np.array([[0, 0], [550, 0], [550, 700], [0, 700]], np.int32)
CONCEALMENT_Y_LINE = 550

# --- POS LOG SIMULATOR ---
# In a real app, this comes from an API. Here, we mock authorized POS triggers.
# Format: [timestamp_in_seconds] that the POS allowed the drawer to open.
MOCK_POS_LOGS = [5.0, 25.0, 60.0]
POS_WINDOW = 10.0  # Drawer is allowed to be open for 10 seconds after a POS command

# --- STATE VARIABLES ---
last_drawer_interaction_time = 0
current_alert = "SECURE"
alert_timer = 0


def check_pos_authorization(current_video_time):
    """Checks if the current time falls within an authorized POS window."""
    for pos_time in MOCK_POS_LOGS:
        if pos_time <= current_video_time <= (pos_time + POS_WINDOW):
            return True
    return False


def process_frame(frame, current_video_time):
    global last_drawer_interaction_time, current_alert, alert_timer

    # Reset alert if timer expires
    if current_video_time > alert_timer:
        current_alert = "SECURE"

    # --- 1. DETECT DRAWER ---
    drawer_results = drawer_model.infer(frame, confidence=0.15)
    drawer_status = "CLOSED"
    drawer_box = None

    for pred in drawer_results[0].predictions:
        if "open" in pred.class_name.lower() or "drawer" in pred.class_name.lower():
            drawer_status = "OPEN"
            x, y, w, h = int(pred.x), int(pred.y), int(
                pred.width), int(pred.height)
            drawer_box = (x - w//2, y - h//2, x + w//2, y + h//2)
            cv2.rectangle(frame, (drawer_box[0], drawer_box[1]),
                          (drawer_box[2], drawer_box[3]), (0, 0, 255), 2)

    # --- 2. DETECT PEOPLE ---
    pose_results = pose_model(frame, verbose=False)
    cashier_present = False

    if pose_results[0].keypoints is not None:
        keypoints = pose_results[0].keypoints.xy.cpu().numpy()
        boxes = pose_results[0].boxes.xyxy.cpu().numpy()

        for i, person_kpts in enumerate(keypoints):
            px_center = int((boxes[i][0] + boxes[i][2]) / 2)
            py_center = int((boxes[i][1] + boxes[i][3]) / 2)

            role = "UNKNOWN"
            if cv2.pointPolygonTest(CASHIER_ZONE, (px_center, py_center), False) >= 0:
                role = "CASHIER"
                cashier_present = True

            # Action Recognition for Cashier (Anomaly 2)
            if role == "CASHIER":
                left_wrist, right_wrist = person_kpts[9], person_kpts[10]

                # Check if hands are in an OPEN drawer
                if drawer_status == "OPEN" and drawer_box is not None:
                    for wrist in [left_wrist, right_wrist]:
                        if wrist[0] > 0 and (drawer_box[0] < wrist[0] < drawer_box[2]) and (drawer_box[1] < wrist[1] < drawer_box[3]):
                            last_drawer_interaction_time = current_video_time

                # Check for Concealment (Hands move below line quickly after drawer access)
                for wrist in [left_wrist, right_wrist]:
                    if wrist[0] > 0 and wrist[1] > CONCEALMENT_Y_LINE:
                        if (current_video_time - last_drawer_interaction_time) < 3.0:
                            current_alert = "ANOMALY 2: SUSPICIOUS CONCEALMENT!"
                            alert_timer = current_video_time + 4.0

    # --- 3. EVALUATE ANOMALIES ---
    is_authorized = check_pos_authorization(current_video_time)

    if drawer_status == "OPEN":
        if not is_authorized:
            # ANOMALY 1: Drawer is open, but no POS log exists
            current_alert = "ANOMALY 1: UNAUTHORIZED OPEN (NO POS LOG)!"
            alert_timer = current_video_time + 2.0

        elif not cashier_present:
            # ANOMALY 3: Drawer is open, but the cashier zone is empty
            current_alert = "ANOMALY 3: UNATTENDED DRAWER!"
            alert_timer = current_video_time + 2.0

    # --- 4. DRAW UI ---
    # Draw zones
    cv2.polylines(frame, [CASHIER_ZONE], True, (255, 0, 255), 2)
    cv2.line(frame, (600, CONCEALMENT_Y_LINE),
             (1200, CONCEALMENT_Y_LINE), (0, 165, 255), 2)

    # Draw Alert Bar
    alert_color = (0, 255, 0) if current_alert == "SECURE" else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (1280, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"STATUS: {current_alert}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 3)

    pos_status = "AUTHORIZED" if is_authorized else "LOCKED"
    cv2.putText(frame, f"POS: {pos_status}", (900, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame


# --- MAIN RUN LOOP ---
video_path = r"C:\Study\HTS\Pet_pooja\cashier1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate current video time in seconds for synchronization
    frame_count += 1
    current_video_time = frame_count / fps

    processed_frame = process_frame(frame, current_video_time)

    # Resize for display if needed
    cv2.imshow("Petpooja Theft Detection System",
               cv2.resize(processed_frame, (1280, 720)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
