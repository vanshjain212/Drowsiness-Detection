import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import csv
from datetime import datetime
import winsound  # Built-in Windows library

# Create or Open the CSV file and write headers if it's new
LOG_FILE = "dms_session_log.csv"

def log_event(event_type, value, threshold):
    """Saves event data to a CSV for post-drive analysis."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event_type, f"{value:.3f}", f"{threshold:.3f}"])
# Initialize CSV with Header (Run once)
with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event", "Value", "Threshold"])

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def ear(landmark, indices, width, height):
    points = []
    for idx in indices:
        x = landmark[idx].x * width
        y = landmark[idx].y * height
        points.append((x, y))
    p1, p2, p3, p4, p5, p6 = points
    v1 = euclidean(p2, p6)
    v2 = euclidean(p3, p5)
    v3 = euclidean(p1, p4)
    return (v1 + v2) / (2.0 * v3)

def mar(landmark, indices, width, height):
    points = []
    for idx in indices:
        x = landmark[idx].x * width
        y = landmark[idx].y * height
        points.append((x, y))
    
    # 8 points for Inner Lips
    p1, p2, p3, p4, p5, p6, p7, p8 = points
    v1 = euclidean(p2, p8)
    v2 = euclidean(p3, p7)
    v3 = euclidean(p4, p6)
    horizontal = euclidean(p1, p5)
    
    # Prevent division by zero just in case
    if horizontal == 0: return 0.0
    return (v1 + v2 + v3) / (2.0 * horizontal)

# --- CONFIGURATION ---
CONSEC_FRAMES = 30
CALIBRATION_FRAMES = 50
YAWN_CONSEC_FRAMES=30

# Landmark Indices (Corrected to Inner Lips for MAR)
left_eye = [362, 385, 387, 263, 373, 380]
right_eye = [33, 160, 158, 133, 153, 144]
mouth = [78, 82, 13, 312, 308, 317, 14, 87]

# --- STATE VARIABLES ---
frame_counter = 0
yawn_counter=0
calib_frame_count = 0
calib_ear_values = []
calib_mar_values = []
eye_alert=False
yawn_alert= False
baseline_ear = 0.0
baseline_mar = 0.0
dynamic_ear_threshold = 0.0
dynamic_mar_threshold = 0.0
is_calibrated = False

# --- MEDIAPIPE INITIALIZATION ---
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)

with vision.FaceLandmarker.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                
                # 1. Calculate Current Ratios
                e_left = ear(face_landmarks, left_eye, width, height)
                e_right = ear(face_landmarks, right_eye, width, height)
                current_ear = (e_left + e_right) / 2.0
                current_mar = mar(face_landmarks, mouth, width, height)

                # ==========================================
                # PHASE 1: DYNAMIC CALIBRATION
                # ==========================================
                if not is_calibrated:
                    cv2.putText(frame, "CALIBRATING... Look straight & neutral", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    calib_ear_values.append(current_ear)
                    calib_mar_values.append(current_mar)
                    calib_frame_count += 1
                    
                    # Draw a loading bar for visual feedback
                    cv2.rectangle(frame, (30, 80), (30 + (calib_frame_count * 8), 100), (0, 255, 255), -1)

                    if calib_frame_count >= CALIBRATION_FRAMES:
                        # Calculate baselines
                        baseline_ear = np.mean(calib_ear_values)
                        baseline_mar = np.mean(calib_mar_values)
                        
                        # Set personalized biological thresholds
                        dynamic_ear_threshold = baseline_ear * 0.75  # 25% drop from resting eye state
                        dynamic_mar_threshold = baseline_mar + 0.55  # Absolute increase from resting mouth state
                        is_calibrated = True
                        print(f"Calibration Complete: EAR Base={baseline_ear:.2f}, MAR Base={baseline_mar:.2f}")

                # ==========================================
                # PHASE 2: ACTIVE MONITORING
                # ==========================================
                else:
                    # --- Drowsy Logic (Eyes) ---
                    if current_ear < dynamic_ear_threshold:
                        frame_counter += 1
                        if frame_counter >= CONSEC_FRAMES and eye_alert==False:
                            cv2.putText(frame, "🚨 DROWSY ALERT! 🚨", (50, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                            
                            # Play high-pitched beep and log data
                            winsound.Beep(1000, 800) 
                            log_event("DROWSY", current_ear, dynamic_ear_threshold)
                            eye_alert=True
                        if frame_counter >= CONSEC_FRAMES:
                            cv2.putText(frame, "🚨 DROWSY ALERT! 🚨", (50, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    else:
                        frame_counter = 0
                        eye_alert=False

                    # --- Yawn Logic (Mouth) ---
                    if current_mar > dynamic_mar_threshold:
                        yawn_counter += 1
                        if yawn_counter >= YAWN_CONSEC_FRAMES and yawn_alert==False:
                            cv2.putText(frame, "⚠️ FATIGUE DETECTED (Yawn) ⚠️", (50, 160), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
                            
                            # Play a lower-pitched beep and log data
                            winsound.Beep(500, 800)
                            log_event("YAWN", current_mar, dynamic_mar_threshold)
                            yawn_alert=True
                        if yawn_counter >= YAWN_CONSEC_FRAMES:
                            cv2.putText(frame, "⚠️ FATIGUE DETECTED (Yawn) ⚠️", (50, 160), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
                    else:
                        yawn_counter = 0
                        yawn_alert=False

                    # UI Dashboard
                    cv2.putText(frame, f"EAR: {current_ear:.2f} (Thresh: {dynamic_ear_threshold:.2f})", 
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"MAR: {current_mar:.2f} (Thresh: {dynamic_mar_threshold:.2f})", 
                                (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Production Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()