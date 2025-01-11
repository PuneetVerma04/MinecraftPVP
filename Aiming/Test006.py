import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
from dataclasses import dataclass

@dataclass
class SensitivitySettings:
    x: float = 6.0
    y: float = 6.0
    acceleration: float = 1.3

class FaceTracker:
    # Key facial landmarks
    NOSE_TIP = 4
    LEFT_EYE = 133
    RIGHT_EYE = 362
    
    def __init__(self, smoothing_frames=3, calibration_frames=20, no_movement_timeout=1.0):
        # Initialize MediaPipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        # Initialize tracking settings
        self.sensitivity = SensitivitySettings()
        self.position_history = deque(maxlen=smoothing_frames)
        self.frame_times = deque(maxlen=30)
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Calibration settings
        self.calibration_center = None
        self.calibration_frames = 0
        self.CALIBRATION_DURATION = calibration_frames
        
        # No movement timeout
        self.NO_MOVEMENT_TIMEOUT = no_movement_timeout
        self.last_movement_time = time.time()
        
        # Disable PyAutoGUI failsafe
        pyautogui.FAILSAFE = False
        
    def calculate_face_center(self, landmarks):
        """Calculate weighted center point between eyes and nose"""
        nose = np.array([landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y])
        left_eye = np.array([landmarks[self.LEFT_EYE].x, landmarks[self.LEFT_EYE].y])
        right_eye = np.array([landmarks[self.RIGHT_EYE].x, landmarks[self.RIGHT_EYE].y])
        
        return nose * 0.7 + (left_eye + right_eye) * 0.15
    
    def smooth_movement(self, current_pos):
        """Apply weighted smoothing to movement"""
        self.position_history.append(current_pos)
        if len(self.position_history) < 2:
            return current_pos
        
        weights = np.array([0.6, 0.3, 0.1][:len(self.position_history)])
        weights = weights / weights.sum()
        positions = np.array(self.position_history)
        
        return np.average(positions, weights=weights, axis=0).astype(int)
    
    def map_to_screen(self, point, frame_width, frame_height):
        """Map face position to screen coordinates with acceleration"""
        x_offset = point[0] - 0.5
        y_offset = point[1] - 0.5
        
        # Apply non-linear acceleration
        x_offset = np.sign(x_offset) * (abs(x_offset) ** self.sensitivity.acceleration)
        y_offset = np.sign(y_offset) * (abs(y_offset) ** self.sensitivity.acceleration)
        
        # Apply sensitivity scaling
        x = 0.5 + (x_offset * self.sensitivity.x)
        y = 0.5 + (y_offset * self.sensitivity.y)
        
        # Map to screen coordinates
        screen_x = int(np.clip(x * self.screen_width, 0, self.screen_width))
        screen_y = int(np.clip(y * self.screen_height, 0, self.screen_height))
        
        return screen_x, screen_y
    
    def process_frame(self, frame):
        """Process a single frame and return tracking results"""
        start_time = time.time()
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB and process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_center = self.calculate_face_center(face_landmarks.landmark)
            
            # Handle calibration
            if self.calibration_frames < self.CALIBRATION_DURATION:
                if self.calibration_center is None:
                    self.calibration_center = face_center
                else:
                    self.calibration_center = self.calibration_center * 0.8 + face_center * 0.2
                self.calibration_frames += 1
                progress = int((self.calibration_frames / self.CALIBRATION_DURATION) * 100)
                cv2.putText(frame, f"Calibrating: {progress}%", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Map and move cursor
                screen_pos = self.map_to_screen(face_center, frame.shape[1], frame.shape[0])
                smoothed_pos = self.smooth_movement(np.array(screen_pos))
                
                # Detect significant movement
                if np.linalg.norm(smoothed_pos - np.array(pyautogui.position())) > 5:  # Significant movement threshold
                    pyautogui.moveTo(*smoothed_pos, _pause=False)
                    self.last_movement_time = time.time()
                else:
                    # Reset cursor if no movement for timeout duration
                    if time.time() - self.last_movement_time > self.NO_MOVEMENT_TIMEOUT:
                        pyautogui.moveTo(self.screen_width // 2, self.screen_height // 2, _pause=False)
                
                # Draw visualization
                center_x = int(face_center[0] * frame.shape[1])
                center_y = int(face_center[1] * frame.shape[0])
                cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0),
                             cv2.MARKER_CROSS, 20, 2)
        
        # Calculate FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        fps = 1 / (sum(self.frame_times) / len(self.frame_times))
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

def main():
    tracker = FaceTracker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Hold your head steady in your preferred center position for calibration...")
    print("Press 'q' to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = tracker.process_frame(frame)
            cv2.imshow('Face Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
