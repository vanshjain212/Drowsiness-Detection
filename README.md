# 🚘 Real-Time Driver Monitoring System (DMS)

An edge-optimized Computer Vision pipeline designed to detect driver fatigue and drowsiness. This project showcases the evolution from computationally heavy Deep Learning classifiers to a lightweight, posture-invariant **Geometric AI** solution.

## 🧠 The Architecture Evolution

This system was built in three distinct phases, prioritizing real-world constraints like demographic bias, edge-device compute limits, and spurious correlations.

### Phase 1: Custom CNN (`/experiments/01_custom_cnn.ipynb`)
* **Approach:** Built a custom Convolutional Neural Network from scratch to classify eye states (Open/Closed).
* **The Engineering Problem:** The model suffered from demographic bias, generalizing poorly across different skin tones and eye morphologies.

### Phase 2: Transfer Learning (`/experiments/02_mobilenetv2.ipynb`)
* **Approach:** Implemented MobileNetV2 to improve feature extraction and accuracy.
* **The Engineering Problem:** Discovered a "shortcut learning" flaw where the model relied on the driver's head-tilt angle rather than actual eyelid closure, resulting in false positives during normal driving movements.

### Phase 3: Geometric AI (Production Build)
* **Approach:** Scrapped the CNN classifiers in favor of **MediaPipe FaceMesh**, extracting 468 3D facial landmarks to calculate deterministic biometric ratios (EAR & MAR).
* **The Result:** A 100% bias-free, explainable, and posture-invariant system capable of running in real-time on low-power edge devices.

## 🚀 Key Features of the Final Build
* **Dynamic Biological Calibration:** The system automatically calculates a personalized resting baseline for every unique driver, completely eliminating morphological bias (e.g., naturally narrow eyes or parted lips).
* **Cross-Sensor Redundancy:** Utilizes the Mouth Aspect Ratio (MAR) as a fallback yawn-detector if the driver's eyes are occluded by sunglasses.
* **Temporal Debouncing:** Implemented custom state variables to prevent haptic/audio alarm spam and UI flickering.
* **Automated Telemetry:** Generates an event-driven `dms_session_log.csv` for post-drive fatigue analysis.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV, MediaPipe Tasks API
* **Mathematics:** NumPy (Linear Algebra / Euclidean Geometry)

## 🏃‍♂️ How to Run the Production Build
1. Clone the repository.
2. Ensure `face_landmarker.task` is in the root directory.
3. Install dependencies: `pip install mediapipe opencv-python numpy`
4. Run the script: `python drowsiness_mediapipe.py`
5. Face the camera for the initial 50-frame biological calibration sequence.