import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pyautogui
from pynput.mouse import Controller
from pynput.keyboard import Controller as KeyboardController
import time  # Import time module for delay

class EnhancedMovementDetector:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Separate thresholds for vertical and horizontal movements
        self.VERTICAL_THRESHOLD = 5    # For forward/backward movements
        self.HORIZONTAL_THRESHOLD = 30  # For left/right movements
        self.SMOOTHING_WINDOW = 5
        
        # Initialize position tracking
        self.position_history = deque(maxlen=self.SMOOTHING_WINDOW)
        self.last_movement = None
        self.movement_cooldown = 0
        self.state = None  # Variable to store movement state

    def calculate_body_reference(self, landmarks):
        """Calculate a more stable reference point using multiple body landmarks"""
        hip_mid = np.array([  # Midpoint of hips
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x) / 2,
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
        ])
        
        shoulder_mid = np.array([  # Midpoint of shoulders
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2,
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        ])
        
        # Use weighted average of hips and shoulders
        return 0.6 * hip_mid + 0.4 * shoulder_mid
    
    def detect_movement(self, frame):
        """Detect movement direction with improved accuracy and different thresholds"""
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)
        
        frame_height, frame_width = frame.shape[:2]
        
        if result.pose_landmarks:
            # Calculate current position using multiple reference points
            current_pos = self.calculate_body_reference(result.pose_landmarks.landmark)
            
            # Convert to pixel coordinates
            current_pos = np.array([  # Current body position in pixel coordinates
                current_pos[0] * frame_width,
                current_pos[1] * frame_height
            ])
            
            # Add to position history
            self.position_history.append(current_pos)
            
            # Only detect movement when we have enough history
            if len(self.position_history) == self.SMOOTHING_WINDOW:
                # Calculate smooth movement vector
                movement_vector = current_pos - self.position_history[0]
                
                # Reduce cooldown
                if self.movement_cooldown > 0:
                    self.movement_cooldown -= 1
                
                # Only detect new movement if cooldown is overws
                if self.movement_cooldown == 0:
                    # Get absolute values for comparison
                    abs_x = abs(movement_vector[0])
                    abs_y = abs(movement_vector[1])
                    
                    # Check if vertical movement exceeds its threshold
                    vertical_significant = abs_y > self.VERTICAL_THRESHOLD
                    # Check if horizontal movement exceeds its threshold
                    horizontal_significant = abs_x > self.HORIZONTAL_THRESHOLD
                    
                    # Determine primary movement direction using different thresholds
                    if vertical_significant and (not horizontal_significant or abs_y > abs_x):
                        if movement_vector[1] < 0:
                            self.state = 0  # Forward
                        else:
                            self.state = 1  # Backward
                    elif horizontal_significant:
                        if movement_vector[0] < 0:
                            self.state = 2  # Left
                        else:
                            self.state = 3  # Right
                    
                    # Set cooldown
                    self.movement_cooldown = 10  # Adjust for sensitivity

            # Draw landmarks and connections
            self.mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return frame

# Game control functions using pyautogui and pynput
def map_action(state, current_actions, keyboard):
    # Continuously press keys based on state
    if state == 0:  # Forward
        if current_actions.get("Forward") != 0:  # Start holding 'W' if not already
            keyboard.press('w')
            current_actions["Forward"] = 0
    elif state == 1:  # Backward
        if current_actions.get("Backward") != 1:  # Start holding 'S' if not already
            keyboard.press('s')
            current_actions["Backward"] = 1
    elif state == 2:  # Right
        if current_actions.get("Right") != 2:  # Start holding 'D' if not already
            keyboard.press('d')
            current_actions["Right"] = 2
    elif state == 3:  # Left
        if current_actions.get("Left") != 3:  # Start holding 'A' if not already
            keyboard.press('a')
            current_actions["Left"] = 3

def stop_action(state, current_actions, keyboard):
    # Release key when state changes
    if state == 0 and current_actions.get("Forward") == 0:
        keyboard.release('w')
        current_actions["Forward"] = None
    elif state == 1 and current_actions.get("Backward") == 1:
        keyboard.release('s')
        current_actions["Backward"] = None
    elif state == 2 and current_actions.get("Right") == 2:
        keyboard.release('d')
        current_actions["Right"] = None
    elif state == 3 and current_actions.get("Left") == 3:
        keyboard.release('a')
        current_actions["Left"] = None

# Main program to detect movements and map to Minecraft actions
def main():
    detector = EnhancedMovementDetector()
    cap = cv2.VideoCapture(0)
    current_actions = {}  # Track current movement actions
    keyboard = KeyboardController()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detector.detect_movement(frame)
        
        if detector.state is not None:
            print(f"Detected movement state: {detector.state}")
            map_action(detector.state, current_actions, keyboard)  # Map detected movement to game controls
            
            # Stop previous movement if the state changes
            if detector.state != detector.last_movement and detector.last_movement is not None:
                stop_action(detector.last_movement, current_actions, keyboard)
            
            # Update last movement
            detector.last_movement = detector.state
        
        cv2.imshow("Enhanced Movement Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Ensure all keys are released when exiting
    for action in current_actions.keys():
        stop_action(action, current_actions, keyboard)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
