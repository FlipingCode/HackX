import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from inference import get_model
from ultralytics import YOLO


st.set_page_config(page_title="Petpooja Smart POS Monitor", layout="wide")
st.title("🛡️ Smart POS Security Dashboard")


@st.cache_resource
def load_models():

    ROBOFLOW_API_KEY = "pAXU8AjF5akbEwwYxx8h"
    drawer_model = get_model(model_id="drawer-alerts/1",
                             api_key=ROBOFLOW_API_KEY)

    pose_model = YOLO('yolov8n-pose.pt')
    return drawer_model, pose_model


drawer_model, pose_model = load_models()

# state
if "ref_frame" not in st.session_state:
    st.session_state.ref_frame = None
if "video_source" not in st.session_state:
    st.session_state.video_source = None

# input
st.sidebar.header("1. Input Source")
input_mode = st.sidebar.radio("Select Video Source:", [
                              " Live Webcam", " Upload Video"])

if input_mode == " Live Webcam":
    if st.sidebar.button("Capture Webcam Setup Frame"):
        cap = cv2.VideoCapture(0)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret:
            st.session_state.ref_frame = frame
            st.session_state.video_source = 0
        cap.release()
        st.rerun()

elif input_mode == " Upload Video":
    uploaded_video = st.sidebar.file_uploader(
        "Upload Station Video", type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        ret, frame = cap.read()
        if ret:
            st.session_state.ref_frame = frame
            st.session_state.video_source = tfile.name
        cap.release()

# zoning
if st.session_state.ref_frame is not None:
    first_frame = st.session_state.ref_frame.copy()
    height, width, _ = first_frame.shape

    st.sidebar.header("2. Draw Zones")
    st.sidebar.markdown("Adjust the sliders to fit the counter layout.")

    st.sidebar.subheader("Cashier Zone (Purple)")
    cash_x1, cash_x2 = st.sidebar.slider(
        "Cashier X", 0, width, (int(width*0.4), width))
    cash_y1, cash_y2 = st.sidebar.slider(
        "Cashier Y", 0, height, (int(height*0.2), height))

    st.sidebar.subheader("Customer Zone (Cyan)")
    cust_x1, cust_x2 = st.sidebar.slider(
        "Customer X", 0, width, (0, int(width*0.4)))
    cust_y1, cust_y2 = st.sidebar.slider(
        "Customer Y", 0, height, (int(height*0.2), height))

    preview_frame = first_frame.copy()
    cv2.rectangle(preview_frame, (cash_x1, cash_y1),
                  (cash_x2, cash_y2), (255, 0, 255), 3)
    cv2.rectangle(preview_frame, (cust_x1, cust_y1),
                  (cust_x2, cust_y2), (255, 255, 0), 3)

    st.markdown(" Zone Configuration Preview")
    st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB),
             use_container_width=True)

    # analysis
    st.sidebar.markdown("---")
    st.sidebar.warning(
        " press q")

    if st.sidebar.button(" Start Fast Analysis", type="primary"):
        st.markdown("---")
        st.markdown("###Live Monitoring Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        drawer_metric = col1.empty()
        cashier_metric = col2.empty()
        customer_metric = col3.empty()
        hand_metric = col4.empty()

        cap = cv2.VideoCapture(st.session_state.video_source)

        # fps
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps == 0 or np.isnan(native_fps):
            native_fps = 30.0
        target_frame_time = 1.0 / native_fps

        if st.session_state.video_source == 0:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_count = 0
        ai_drawer_status = "CLOSED"
        detected_drawer_box = None
        debug_predictions = []

        while cap.isOpened():
            loop_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                st.success(" Video playback complete.")
                break

            frame_count += 1

            # drawe
            if frame_count % 3 == 0:

                drawer_results = drawer_model.infer(frame, confidence=0.01)
                ai_drawer_status = "CLOSED"
                detected_drawer_box = None

                debug_predictions = drawer_results[0].predictions

                for pred in debug_predictions:
                    if "open" in pred.class_name.lower() or "drawer" in pred.class_name.lower():
                        ai_drawer_status = "OPEN"
                        x, y, w, h = int(pred.x), int(pred.y), int(
                            pred.width), int(pred.height)
                        detected_drawer_box = (
                            x - w//2, y - h//2, x + w//2, y + h//2)
                        break

            # people
            pose_results = pose_model(frame, verbose=False)
            annotated_frame = pose_results[0].plot()

            for pred in debug_predictions:
                px, py, pw, ph = int(pred.x), int(
                    pred.y), int(pred.width), int(pred.height)
                cv2.rectangle(annotated_frame, (px - pw//2, py - ph//2),
                              (px + pw//2, py + ph//2), (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"{pred.class_name} {pred.confidence:.2f}", (
                    px - pw//2, py - ph//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cashier_count, customer_count = 0, 0
            hands_in_drawer = "NO"

            if pose_results[0].keypoints is not None:
                keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                boxes = pose_results[0].boxes.xyxy.cpu().numpy()

                for i, kpts in enumerate(keypoints):
                    px_center = int((boxes[i][0] + boxes[i][2]) / 2)
                    py_center = int((boxes[i][1] + boxes[i][3]) / 2)
                    role = "UNKNOWN"

                    if cash_x1 < px_center < cash_x2 and cash_y1 < py_center < cash_y2:
                        role = "CASHIER"
                        cashier_count += 1
                        color = (255, 0, 255)
                    elif cust_x1 < px_center < cust_x2 and cust_y1 < py_center < cust_y2:
                        role = "CUSTOMER"
                        customer_count += 1
                        color = (255, 255, 0)
                    else:
                        color = (128, 128, 128)

                    cv2.putText(annotated_frame, role, (int(boxes[i][0]), int(boxes[i][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    for wrist in [kpts[9], kpts[10]]:
                        if wrist[0] > 0 and wrist[1] > 0:
                            wx, wy = int(wrist[0]), int(wrist[1])
                            cv2.circle(annotated_frame, (wx, wy),
                                       8, (0, 255, 0), -1)

                            if detected_drawer_box is not None:
                                dx1, dy1, dx2, dy2 = detected_drawer_box
                                if dx1 < wx < dx2 and dy1 < wy < dy2:
                                    hands_in_drawer = f"YES ({role})"
                                    cv2.circle(annotated_frame,
                                               (wx, wy), 15, (0, 0, 255), 3)
                                    cv2.putText(annotated_frame, "HAND IN DRAWER", (wx - 50, wy - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(annotated_frame, (cash_x1, cash_y1),
                          (cash_x2, cash_y2), (255, 0, 255), 2)
            cv2.rectangle(annotated_frame, (cust_x1, cust_y1),
                          (cust_x2, cust_y2), (255, 255, 0), 2)

            if detected_drawer_box is not None and ai_drawer_status == "OPEN":
                dx1, dy1, dx2, dy2 = detected_drawer_box
                cv2.rectangle(annotated_frame, (dx1, dy1),
                              (dx2, dy2), (0, 0, 255), 3)
                cv2.putText(annotated_frame, f"AI: OPEN DRAWER", (dx1,
                            dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            drawer_color = "normal" if ai_drawer_status == "CLOSED" else "inverse"
            drawer_metric.metric("Drawer State (AI)", ai_drawer_status,
                                 delta="Alert" if ai_drawer_status == "OPEN" else "Secure", delta_color=drawer_color)
            cashier_metric.metric("Cashiers Present", cashier_count)
            customer_metric.metric("Customers Present", customer_count)
            hand_metric.metric("Hands in Drawer?", hands_in_drawer, delta="Active" if "YES" in hands_in_drawer else "Clear",
                               delta_color="inverse" if "YES" in hands_in_drawer else "normal")

            cv2.imshow(
                "Petpooja Security Feed - LIVE (Press 'Q' to Quit)", annotated_frame)

            processing_time = time.time() - loop_start_time
            sleep_time_ms = max(
                1, int((target_frame_time - processing_time) * 1000))

            if cv2.waitKey(sleep_time_ms) & 0xFF == ord('q'):
                st.warning("Analysis manually stopped.")
                break

        cap.release()
        cv2.destroyAllWindows()
else:
    st.info(" begin.")
