import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
from collections import deque
import mediapipe as mp

class FatigueCalculator:
    def __init__(self):
        # Configuration
        self.EAR_THRESHOLD_SENSITIVITY = 0.80  
        self.EAR_CONSEC_FRAMES = 3             
        self.MAR_THRESHOLD_SENSITIVITY = 2.5   
        self.MAR_CONSEC_FRAMES = 8             
        self.CALIBRATION_DURATION = 20         
        self.PERCLOS_WINDOW_SECONDS = 60

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # State variables
        self.reset()

    def reset(self):
        self.is_calibrating = False
        self.calibration_start_time = 0
        self.calibration_ear_data = []
        self.calibration_mar_data = []
        self.baseline_ear = 0.3
        self.baseline_mar = 0.1
        self.ear_threshold = 0.23
        self.mar_threshold = 0.5
        
        self.blink_counter = 0
        self.is_blinking = False
        self.total_blinks = 0
        
        self.yawn_counter = 0
        self.is_yawning = False
        self.total_yawns = 0

        self.perclos_history = deque(maxlen=self.PERCLOS_WINDOW_SECONDS * 20)
        self.fatigue_index = 0.0

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[2], mouth[10])  
        B = dist.euclidean(mouth[4], mouth[8])   
        C = dist.euclidean(mouth[0], mouth[6])   
        mar = (A + B) / (2.0 * C)
        return mar

    def start_calibration(self):
        print("[INFO] Starting calibration. Please look straight and maintain a neutral expression.")
        self.reset()
        self.is_calibrating = True
        self.calibration_start_time = time.time()

    def _finish_calibration(self):
        if not self.calibration_ear_data or not self.calibration_mar_data:
            print("[WARNING] Calibration failed: not enough data. Using default values.")
            return

        self.baseline_ear = np.mean(self.calibration_ear_data)
        self.baseline_mar = np.mean(self.calibration_mar_data)
        
        self.ear_threshold = self.baseline_ear * self.EAR_THRESHOLD_SENSITIVITY
        self.mar_threshold = self.baseline_mar * self.MAR_THRESHOLD_SENSITIVITY

        self.is_calibrating = False
        print("[INFO] Calibration complete.")
        print(f" -> Baseline EAR: {self.baseline_ear:.3f}, Threshold: {self.ear_threshold:.3f}")
        print(f" -> Baseline MAR: {self.baseline_mar:.3f}, Threshold: {self.mar_threshold:.3f}")

    def get_fatigue_index(self):
        """
        Returns fatigue index in [0, 1] — 0 = fully alert, 1 = extremely fatigued.
        Scaled to be more sensitive for real-time UI feedback.
        """
        perclos = np.mean(self.perclos_history) if self.perclos_history else 0
        
        # Estimate blinks per minute for normalization
        if self.perclos_history:
            duration_minutes = len(self.perclos_history) / (20 * 60)  
            blink_rate_per_min = self.total_blinks / max(duration_minutes, 0.1)  
        else:
            blink_rate_per_min = 0

        normalized_blinks = min(blink_rate_per_min / 30.0, 1.0)  
        normalized_yawns = min(self.total_yawns / 3.0, 1.0)      

        raw_fatigue = 0.6 * perclos + 0.25 * normalized_blinks + 0.15 * normalized_yawns
        self.fatigue_index = min(1.0, raw_fatigue * 3.33)

        return round(self.fatigue_index, 4)

    def update_metrics(self, frame):
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        current_ear = self.baseline_ear
        current_mar = self.baseline_mar
        
        if results.multi_face_landmarks:
            landmarks_list = results.multi_face_landmarks[0].landmark
            face_landmarks = [[lm.x * w, lm.y * h] for lm in landmarks_list]
            face_landmarks = np.array(face_landmarks)

            l_eye = face_landmarks[[362, 385, 387, 263, 373, 380]] 
            r_eye = face_landmarks[[33, 160, 158, 133, 153, 144]]  
            mouth = face_landmarks[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]]

            current_ear = (self.eye_aspect_ratio(l_eye) + self.eye_aspect_ratio(r_eye)) / 2.0
            current_mar = self.mouth_aspect_ratio(mouth)

            if self.is_calibrating:
                if time.time() - self.calibration_start_time < self.CALIBRATION_DURATION:
                    self.calibration_ear_data.append(current_ear)
                    self.calibration_mar_data.append(current_mar)
                else:
                    self._finish_calibration()
            else:
                # Detect blinking
                if current_ear < self.ear_threshold:
                    self.blink_counter += 1
                    self.perclos_history.append(1)
                else:
                    if self.blink_counter >= self.EAR_CONSEC_FRAMES:
                        self.total_blinks += 1
                    self.blink_counter = 0
                    self.perclos_history.append(0)

                # Detect yawning
                if current_mar > self.mar_threshold:
                    self.yawn_counter += 1
                else:
                    if self.yawn_counter >= self.MAR_CONSEC_FRAMES:
                        self.total_yawns += 1
                    self.yawn_counter = 0
        
        self.is_blinking = self.blink_counter > 0
        self.is_yawning = self.yawn_counter > 0

        return {
            'ear': round(current_ear, 4),
            'mar': round(current_mar, 4),
            'is_blinking': self.is_blinking,
            'is_yawning': self.is_yawning
        }