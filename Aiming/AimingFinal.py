import cv2
import mediapipe as mp
import pynput.mouse as mouse
import numpy as np
import time
from collections import deque
from dataclasses import dataclass

@dataclass
class SensitivitySettings:
    x: float = 2.0  # Reduced for more precise control
    y: float = 1.5
    acceleration: float = 1.2
    dead_zone: float = 0.02  # Minimum movement threshold

class FaceTracker:
    NOSE_TIP = 4
    LEFT_EYE = 133
    RIGHT_EYE = 362
    
    def __init__(self, smoothing_frames=3, calibration_frames=20):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        self.sensitivity = SensitivitySettings()
        self.position_history = deque(maxlen=smoothing_frames)
        self.frame_times = deque(maxlen=30)
        
        # Initialize mouse controller for relative movement
        self.mouse = mouse.Controller()
        
        # Calibration and tracking
        self.calibration_center = None
        self.calibration_frames = 0
        self.CALIBRATION_DURATION = calibration_frames
        self.last_position = None
        
    def calculate_face_center(self, landmarks):
        """Calculate weighted center point between eyes and nose"""
        nose = np.array([landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y])
        left_eye = np.array([landmarks[self.LEFT_EYE].x, landmarks[self.LEFT_EYE].y])
        right_eye = np.array([landmarks[self.RIGHT_EYE].x, landmarks[self.RIGHT_EYE].y])
        
        return nose * 0.7 + (left_eye + right_eye) * 0.15
    
    def calculate_relative_movement(self, current_pos):
        """Calculate relative mouse movement from face position"""
        if self.last_position is None:
            self.last_position = current_pos
            return 0, 0
        
        # Calculate offset from calibration center
        x_offset = (current_pos[0] - self.calibration_center[0])
        y_offset = (current_pos[1] - self.calibration_center[1])
        
        # Apply dead zone
        if abs(x_offset) < self.sensitivity.dead_zone:
            x_offset = 0
        if abs(y_offset) < self.sensitivity.dead_zone:
            y_offset = 0
            
        # Apply non-linear acceleration
        x_movement = np.sign(x_offset) * (abs(x_offset) ** self.sensitivity.acceleration)
        y_movement = np.sign(y_offset) * (abs(y_offset) ** self.sensitivity.acceleration)
        
        # Scale by sensitivity
        x_movement *= self.sensitivity.x * 100  # Scale up for more noticeable movement
        y_movement *= self.sensitivity.y * 100
        
        # Update last position
        self.last_position = current_pos
        
        return int(x_movement), int(y_movement)
    
    def process_frame(self, frame):
        """Process a single frame and control mouse movement"""
        start_time = time.time()
        
        frame = cv2.flip(frame, 1)
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
                # Calculate and apply relative movement
                dx, dy = self.calculate_relative_movement(face_center)
                if dx != 0 or dy != 0:
                    self.mouse.move(dx, dy)
                
                # Visualization
                center_x = int(face_center[0] * frame.shape[1])
                center_y = int(face_center[1] * frame.shape[0])
                cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0),
                             cv2.MARKER_CROSS, 20, 2)
                
                # Draw movement vector
                if self.calibration_center is not None:
                    calib_x = int(self.calibration_center[0] * frame.shape[1])
                    calib_y = int(self.calibration_center[1] * frame.shape[0])
                    cv2.line(frame, (calib_x, calib_y), (center_x, center_y),
                            (255, 0, 0), 2)
        
        # Calculate and display FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        fps = 1 / (sum(self.frame_times) / len(self.frame_times))
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

def main():
    print("\nGame Camera Control Setup:")
    print("1. Launch your game in windowed or fullscreen mode")
    print("2. Make sure the game window is in focus")
    print("3. Hold your head steady for initial calibration")
    print("4. Move your head slightly to control the camera")
    print("5. Press 'q' to exit\n")
    
    tracker = FaceTracker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = tracker.process_frame(frame)
            cv2.imshow('Game Camera Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()