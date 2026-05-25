import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import json
import tempfile
import os
import pandas as pd
import datetime
import time
# --- IMPORT MODULAR LOGIC ---
from detection import calculate_ear, calculate_mar

# --- PAGE SETUP ---
st.set_page_config(page_title="Driver Fatigue Analytics", page_icon="🚘", layout="wide")
st.title("🚘 Edge-Optimized Driver Monitoring System")
st.markdown("Analyze driver dashcam footage for fatigue events using Geometric AI.")

# --- LOAD DEFAULTS FROM CONFIG ---
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"EAR_THRESHOLD": 0.22, "MAR_THRESHOLD": 0.55, "CONSEC_FRAMES": 30}

# --- SIDEBAR CONFIGURATION ---
with st.sidebar.expander("⚙️ System Threshold Configuration", expanded=False):
    st.markdown("Adjust these sliders to tune the mathematics and eliminate false positives.")
    ear_threshold = st.slider("Eye Aspect Ratio (EAR) Threshold", 0.15, 0.35, float(config.get("EAR_THRESHOLD", 0.22)), 0.01)
    mar_threshold = st.slider("Mouth Aspect Ratio (MAR) Threshold", 0.40, 0.80, float(config.get("MAR_THRESHOLD", 0.55)), 0.01)
    consec_frames = st.slider("Drowsiness Frame Window", 10, 60, int(config.get("CONSEC_FRAMES", 30)), 5)

# --- LANDMARK INDICES ---
left_eye = [362, 385, 387, 263, 373, 380]
right_eye = [33, 160, 158, 133, 153, 144]
mouth = [78, 82, 13, 312, 308, 317, 14, 87]

# --- THE ZERO-FRICTION UI ---
st.subheader("Select a Demo or Upload Your Own")

demo_option = st.selectbox(
    "Choose a test video:",
    ("None", "Demo: Normal Driving", "Demo: Yawning", "Demo: Drowsy (Eyes Closed)", "Upload my own video")
)

# Determine the video source
video_path = None
uploaded_file = None

if demo_option == "Demo: Normal Driving":
    video_path = "data/test_alert.mp4"
elif demo_option == "Demo: Yawning":
    video_path = "data/test_yawn.mp4"
elif demo_option == "Demo: Drowsy (Eyes Closed)":
    video_path = "data/test_drowsy.mp4"
elif demo_option == "Upload my own video":
    uploaded_file = st.file_uploader("Upload an .mp4", type=["mp4", "mov", "avi"])

# --- PROCESSING PIPELINE ---
if video_path is not None or uploaded_file is not None:
    
    # Handle file extraction
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        target_video = tfile.name
    else:
        target_video = video_path

    # Check if the demo file actually exists
    if not os.path.exists(target_video) and uploaded_file is None:
        st.error(f"Error: Could not find '{target_video}'. Please ensure the video is in the 'data/' folder.")
    else:
        st.markdown("---")
        col_video, col_metrics = st.columns([2, 1])
        
        with col_video:
            st.subheader("Live Video Feed")
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
        with col_metrics:
            st.subheader("Real-Time Telemetry")
            metric_drowsy = st.metric("Drowsy Events Detected", "0")
            metric_yawn = st.metric("Yawn Events Detected", "0")
            metric_risk = st.metric("Fatigue Risk Score", "0%")
        
        # --- STATE VARIABLES & DEBOUNCING ---
        drowsy_frames = 0
        total_yawn_alerts = 0
        total_drowsy_alerts = 0
        yawn_cooldown = 0   # Cooldown timer to prevent double-counting yawns
        drowsy_flag = False
        
        # --- DATA SEPARATION ---
        session_telemetry = [] # High-frequency data for UI charts
        event_log = []         # Low-frequency data for CSV export
        frame_count = 0

        base_options = python.BaseOptions(
            model_asset_path='face_landmarker.task',
            delegate=python.BaseOptions.Delegate.CPU
            )        
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        
        with vision.FaceLandmarker.create_from_options(options) as detector:
            cap = cv2.VideoCapture(target_video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)
                
                current_ear = 0.0
                current_mar = 0.0
                current_event = "Normal"
                
                if detection_result.face_landmarks:
                    for face_landmarks in detection_result.face_landmarks:
                        e_l = calculate_ear(face_landmarks, left_eye, w, h)
                        e_r = calculate_ear(face_landmarks, right_eye, w, h)
                        current_ear = (e_l + e_r) / 2.0
                        current_mar = calculate_mar(face_landmarks, mouth, w, h)
                        
                        # --- 1. DROWSY LOGIC ---
                        if current_ear < ear_threshold:
                            drowsy_frames += 1
                            if drowsy_frames >= consec_frames:
                                cv2.putText(frame, "ALERT: DROWSY", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                                current_event = "Drowsy"
                                
                                if not drowsy_flag:
                                    total_drowsy_alerts += 1
                                    drowsy_flag = True
                                    
                                    # Log the discrete event for the CSV
                                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    event_log.append({"Timestamp": timestamp, "Event": "DROWSY", "Value": current_ear, "Threshold": ear_threshold})
                        else:
                            drowsy_frames = 0
                            drowsy_flag = False
                            
                        # --- 2. YAWN LOGIC (WITH DEBOUNCING) ---
                        if yawn_cooldown > 0:
                            yawn_cooldown -= 1
                            
                        if current_mar > mar_threshold:
                            cv2.putText(frame, "WARNING: YAWNING", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 3)
                            if current_event == "Normal": 
                                current_event = "Yawn"
                                
                            if yawn_cooldown == 0:
                                total_yawn_alerts += 1
                                yawn_cooldown = 45 # Lock out other yawns for ~1.5 seconds
                                
                                # Log the discrete event for the CSV
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                event_log.append({"Timestamp": timestamp, "Event": "YAWN", "Value": current_mar, "Threshold": mar_threshold})

                        # Calculate dynamic risk score
                        risk_score = min(100, (total_yawn_alerts * 15) + (total_drowsy_alerts * 35))

                        # Update UI Metrics dynamically
                        metric_drowsy.metric("Drowsy Events Detected", str(total_drowsy_alerts))
                        metric_yawn.metric("Yawn Events Detected", str(total_yawn_alerts))
                        metric_risk.metric("Fatigue Risk Score", f"{risk_score}%")

                # Log high-frequency data for the charts
                session_telemetry.append({
                    "Frame": frame_count,
                    "EAR": current_ear,
                    "MAR": current_mar,
                    "Event": current_event
                })

                # --- PREVENT WEBSOCKET FLOODING ---
                # Only send every 3rd frame to the web browser to keep the video playing smoothly
                # --- HIGH-EFFICIENCY GRAPHICS RENDERING ---
                # 1. Skip visual rendering for 2 out of every 3 frames to clear the WebSocket queue
                if frame_count % 3 == 0:
                    # 2. Downscale frame resolution slightly to massively decrease internet payload size
                    preview_h, preview_w, _ = frame.shape
                    preview_resized = cv2.resize(frame, (int(preview_w * 0.7), int(preview_h * 0.7)))
                    
                    # 3. Convert and send to the container player
                    rgb_preview = cv2.cvtColor(preview_resized, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_preview, channels="RGB", use_container_width=True)
                    
                    # 4. Give the React engine a tiny breathing window to repaint the browser page
                    time.sleep(0.03)
                
            cap.release()
            
        # --- POST-PROCESSING ANALYTICS DASHBOARD ---
        st.success("✅ Analysis Complete!")
        st.markdown("---")
        st.subheader("📊 Post-Drive Session Analytics")
        
        # Convert frame-by-frame memory list to Pandas DataFrame for Charts
        chart_df = pd.DataFrame(session_telemetry)
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("**Biometric Ratios Over Time**")
            st.line_chart(chart_df.set_index("Frame")[["EAR", "MAR"]])
            
        with chart_col2:
            st.markdown("**Event Frequency**")
            event_counts = chart_df[chart_df["Event"] != "Normal"]["Event"].value_counts()
            if not event_counts.empty:
                st.bar_chart(event_counts, color="#ff4b4b")
            else:
                st.info("No fatigue events detected during this session.")
        
        # --- THE CLEAN CSV EXPORT ---
        event_df = pd.DataFrame(event_log)
        if event_df.empty:
            # Ensure columns exist even if no events happened
            event_df = pd.DataFrame(columns=["Timestamp", "Event", "Value", "Threshold"])
            
        csv_data = event_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Fatigue Event Log (CSV)",
            data=csv_data,
            file_name="dms_fatigue_events.csv",
            mime="text/csv",
        )

        # Clean up temporary file if uploaded
        if uploaded_file is not None:
            os.remove(target_video)