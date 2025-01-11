import cv2
import mediapipe as mp
import numpy as np
import time
from dataclasses import dataclass
import pydirectinput
import pyautogui

@dataclass
class GameSettings:
    x_sensitivity: float = 8.0  # Increased for game movement
    y_sensitivity: float = 6.0
    acceleration: float = 1.2
    calibration_frames: int = 30
    no_movement_threshold: float = 2.0  # Pixels of movement required

class MinecraftFaceController:
    # Key facial landmarks
    NOSE_TIP = 4
    LEFT_EYE = 133
    RIGHT_EYE = 362
    
    def __init__(self):
        # Initialize MediaPipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        # Game settings
        self.settings = GameSettings()
        
        # Initialize tracking variables
        self.previous_pos = None
        self.calibration_center = None
        self.calibration_frames = 0
        
        # Configure direct input for game control
        pydirectinput.PAUSE = 0
        
    def calculate_face_center(self, landmarks):
        """Calculate weighted center point between eyes and nose"""
        nose = np.array([landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y])
        left_eye = np.array([landmarks[self.LEFT_EYE].x, landmarks[self.LEFT_EYE].y])
        right_eye = np.array([landmarks[self.RIGHT_EYE].x, landmarks[self.RIGHT_EYE].y])
        
        return nose * 0.7 + (left_eye + right_eye) * 0.15
    
    def calculate_relative_movement(self, current_pos):
        """Calculate relative mouse movement based on face position"""
        if self.previous_pos is None:
            self.previous_pos = current_pos
            return 0, 0
            
        # Calculate relative movement
        x_movement = (current_pos[0] - self.calibration_center[0]) * self.settings.x_sensitivity
        y_movement = (current_pos[1] - self.calibration_center[1]) * self.settings.y_sensitivity
        
        # Apply non-linear acceleration for more precise control
        x_movement = np.sign(x_movement) * (abs(x_movement) ** self.settings.acceleration)
        y_movement = np.sign(y_movement) * (abs(y_movement) ** self.settings.acceleration)
        
        # Convert to integers for mouse movement
        x_movement = int(x_movement)
        y_movement = int(y_movement)
        
        return x_movement, y_movement
    
    def process_frame(self, frame):
        """Process a single frame and control game aiming"""
        # Flip frame horizontally for more intuitive control
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
            if self.calibration_frames < self.settings.calibration_frames:
                if self.calibration_center is None:
                    self.calibration_center = face_center
                else:
                    self.calibration_center = self.calibration_center * 0.8 + face_center * 0.2
                self.calibration_frames += 1
                progress = int((self.calibration_frames / self.settings.calibration_frames) * 100)
                cv2.putText(frame, f"Calibrating: {progress}%", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Calculate and apply movement directly
                x_move, y_move = self.calculate_relative_movement(face_center)
                
                # Only move if above threshold
                if abs(x_move) > self.settings.no_movement_threshold or \
                   abs(y_move) > self.settings.no_movement_threshold:
                    pydirectinput.move(x_move, y_move)
                
                # Visualize tracking point
                center_x = int(face_center[0] * frame.shape[1])
                center_y = int(face_center[1] * frame.shape[0])
                cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0),
                             cv2.MARKER_CROSS, 20, 2)
        
        return frame

def main():
    controller = MinecraftFaceController()
    cap = cv2.VideoCapture(0)
    
    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Hold your head steady in your preferred center position...")
    print("Press 'q' to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = controller.process_frame(frame)
            cv2.imshow('Minecraft Face Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()