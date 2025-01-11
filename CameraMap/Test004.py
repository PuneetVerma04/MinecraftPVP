import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
import win32api
import win32con
import keyboard

@dataclass
class SensitivitySettings:
    x: float = 8.0  # Reduced for better control
    y: float = 8.0  # Slightly lower vertical sensitivity
    acceleration: float = 1.0  # Reduced for more linear response
    dead_zone: float = 0.03  # Increased dead zone for stability
    movement_scale: float = 30.0  # Adjusted for smoother movement
    max_movement: float = 40.0  # Maximum movement threshold

class FaceTracker:
    NOSE_TIP = 4
    LEFT_EYE = 133
    RIGHT_EYE = 362
    
    def __init__(self, smoothing_frames=3, calibration_frames=30):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        self.sensitivity = SensitivitySettings()
        self.position_history = deque(maxlen=smoothing_frames)
        self.frame_times = deque(maxlen=30)
        
        # Calibration and tracking
        self.calibration_center = None
        self.calibration_frames = 0
        self.CALIBRATION_DURATION = calibration_frames
        self.last_position = None
        self.is_calibrating = False
        
    def recenter_view(self, current_pos):
        """Recenter the view and recalibrate"""
        self.calibration_center = current_pos
        # Reset tracking variables
        self.last_position = current_pos
        self.position_history.clear()
        return "Recentered"
        
    def calculate_face_center(self, landmarks):
        """Calculate weighted center point between eyes and nose"""
        nose = np.array([landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y])
        left_eye = np.array([landmarks[self.LEFT_EYE].x, landmarks[self.LEFT_EYE].y])
        right_eye = np.array([landmarks[self.RIGHT_EYE].x, landmarks[self.RIGHT_EYE].y])
        
        return nose * 0.7 + (left_eye + right_eye) * 0.15
    
    def move_camera(self, dx: int, dy: int):
        """Send direct mouse movement input with clamping"""
        # Clamp movement to maximum values
        dx = np.clip(dx, -self.sensitivity.max_movement, self.sensitivity.max_movement)
        dy = np.clip(dy, -self.sensitivity.max_movement, self.sensitivity.max_movement)
        
        if abs(dx) > 0 or abs(dy) > 0:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
    
    def calculate_relative_movement(self, current_pos):
        """Calculate relative camera movement from face position"""
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
        
        # Scale by sensitivity and movement scale
        x_movement *= self.sensitivity.x * self.sensitivity.movement_scale
        y_movement *= self.sensitivity.y * self.sensitivity.movement_scale
        
        # Update last position
        self.last_position = current_pos
        
        return x_movement, y_movement
    
    def process_frame(self, frame):
        """Process a single frame and control camera movement"""
        start_time = time.time()
        status_message = ""
        
        # Check for recenter command
        if keyboard.is_pressed('r'):
            self.is_calibrating = True
            status_message = "Recentering..."
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_center = self.calculate_face_center(face_landmarks.landmark)
            
            # Handle initial calibration or recentering
            if self.calibration_frames < self.CALIBRATION_DURATION or self.is_calibrating:
                if self.calibration_center is None or self.is_calibrating:
                    self.calibration_center = face_center
                    self.is_calibrating = False
                else:
                    self.calibration_center = self.calibration_center * 0.8 + face_center * 0.2
                self.calibration_frames += 1
                progress = int((self.calibration_frames / self.CALIBRATION_DURATION) * 100)
                status_message = f"Calibrating: {progress}%"
            else:
                # Calculate and apply camera movement
                dx, dy = self.calculate_relative_movement(face_center)
                if dx != 0 or dy != 0:
                    self.move_camera(dx, dy)
                status_message = f"Move: {dx:.1f}, {dy:.1f}"
                
                # Draw tracking visualization
                center_x = int(face_center[0] * frame.shape[1])
                center_y = int(face_center[1] * frame.shape[0])
                cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0),
                             cv2.MARKER_CROSS, 20, 2)
                
                if self.calibration_center is not None:
                    calib_x = int(self.calibration_center[0] * frame.shape[1])
                    calib_y = int(self.calibration_center[1] * frame.shape[0])
                    cv2.line(frame, (calib_x, calib_y), (center_x, center_y),
                            (255, 0, 0), 2)
        
        # Calculate and display FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        fps = 1 / (sum(self.frame_times) / len(self.frame_times))
        
        # Draw status information
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status_message, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'R' to recenter", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

def main():
    print("\nGame Camera Control Setup:")
    print("1. Launch your game and enter the game world")
    print("2. Make sure mouse cursor is hidden/locked in game")
    print("3. Hold your head steady for initial calibration")
    print("4. Use 'R' key to recenter the view at any time")
    print("5. Press 'Q' to exit\n")
    
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