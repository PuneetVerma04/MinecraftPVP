import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import win32api
import win32con

@dataclass
class GameConfig:
    """Configuration settings for the game detection"""
    FPS: int = 30
    HISTORY_SIZE: int = 45  # 1.5 seconds at 30 FPS
    VERTICAL_MOVEMENT_THRESHOLD: float = 0.03
    SWING_STATE_THRESHOLD: int = 2
    COOLDOWN_FRAMES: int = 8
    SHOULDER_HEIGHT_THRESHOLD: float = 0.05
    SHIELD_BUFFER_SIZE: int = 5

class SwingDetector:
    """Handles detection of sword swing motions using pose landmarks"""
    def __init__(self, config: GameConfig):
        self.config = config
        self.landmark_history = deque(maxlen=config.HISTORY_SIZE)
        self.swing_state = "ready"
        self.state_counter = 0
        self.total_swings = 0
        self.cooldown_counter = 0
        self.last_click_state = False  # Track if we've already clicked for this swing
    
    def perform_attack_click(self):
        """Perform left mouse click for attack"""
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self.last_click_state = True

    def reset_click_state(self):
        """Reset the click state when returning to ready"""
        self.last_click_state = False
    
    def get_wrist_position(self, landmarks: List[List[float]]) -> float:
        """Calculate relative position of right wrist to right shoulder"""
        if not landmarks:
            return 0
        right_shoulder = np.array([landmarks[12][1]])
        right_wrist = np.array([landmarks[16][1]])
        return float(right_shoulder - right_wrist)

    def detect_swing(self, landmarks: List[List[float]], shield_state: str) -> bool:
        """
        Detect if a swing motion has occurred with the right hand
        Returns to ready state if shield is up
        """
        self.landmark_history.append(landmarks)
        
        # If shield is up, reset to ready state and block swings
        if shield_state == "Up":
            self.swing_state = "ready"
            self.state_counter = 0
            self.reset_click_state()
            return False
        
        if len(self.landmark_history) < 2 or self.cooldown_counter > 0:
            self.cooldown_counter = max(0, self.cooldown_counter - 1)
            return False
            
        current_pos = self.get_wrist_position(self.landmark_history[-1])
        prev_pos = self.get_wrist_position(self.landmark_history[-2])
        pos_change = current_pos - prev_pos
        
        # State machine for swing detection
        swing_detected = False
        if self.swing_state == "ready":
            self.reset_click_state()
            if pos_change > self.config.VERTICAL_MOVEMENT_THRESHOLD:
                self.state_counter += 1
                if self.state_counter >= self.config.SWING_STATE_THRESHOLD:
                    self.swing_state = "lifting"
                    self.state_counter = 0
            else:
                self.state_counter = 0
                
        elif self.swing_state == "lifting":
            if -self.config.SHOULDER_HEIGHT_THRESHOLD <= current_pos <= 0.2:
                if pos_change < -self.config.VERTICAL_MOVEMENT_THRESHOLD:
                    self.swing_state = "swinging"
                    self.state_counter = 0
            else:
                self.swing_state = "ready"
                self.state_counter = 0
                
        elif self.swing_state == "swinging":
            if pos_change < -self.config.VERTICAL_MOVEMENT_THRESHOLD:
                self.state_counter += 1
                if self.state_counter >= self.config.SWING_STATE_THRESHOLD:
                    if not self.last_click_state:  # Only click if we haven't already
                        self.perform_attack_click()
                    self.swing_state = "ready"
                    self.state_counter = 0
                    self.total_swings += 1
                    self.cooldown_counter = self.config.COOLDOWN_FRAMES
                    swing_detected = True
            else:
                self.state_counter = 0
        
        return swing_detected

    # ... (rest of SwingDetector methods remain the same)

class ShieldDetector:
    """Handles detection of shield (stop gesture) using left hand landmarks"""
    def __init__(self, config: GameConfig):
        self.config = config
        self.shield_buffer = [False] * config.SHIELD_BUFFER_SIZE
        self.shield_state = "Down"
        self.last_shield_state = "Down"
    
    @staticmethod
    def is_left_hand(handedness) -> bool:
        """
        Check if the detected hand is the left hand (in mirrored view)
        When using front camera, "Right" label actually means it's our left hand
        """
        return handedness.classification[0].label == "Right"
    
    def perform_shield_click(self, shield_up: bool):
        """Perform right mouse click for shield"""
        if shield_up and self.last_shield_state == "Down":
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        elif not shield_up and self.last_shield_state == "Up":
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    
    def is_stop_gesture(self, landmarks) -> bool:
        """Detect if the hand is making a stop gesture"""
        points = {
            'wrist': landmarks[0],
            'thumb_tip': landmarks[4],
            'index': {'tip': landmarks[8], 'pip': landmarks[6]},
            'middle': {'tip': landmarks[12], 'pip': landmarks[10]},
            'ring': {'tip': landmarks[16], 'pip': landmarks[14]},
            'pinky': {'tip': landmarks[20], 'pip': landmarks[18]}
        }
        
        finger_extended = [
            points[finger]['tip'].y < points[finger]['pip'].y
            for finger in ['index', 'middle', 'ring', 'pinky']
        ]
        
        fingertips = [points[finger]['tip'] for finger in ['index', 'middle', 'ring', 'pinky']]
        hand_raised = all(tip.y < points['wrist'].y for tip in fingertips)
        
        palm_facing = points['wrist'].z > landmarks[9].z
        
        return all(finger_extended) and hand_raised and palm_facing

    def update_shield_state(self, current_shield_up: bool) -> str:
        """Update and return the current shield state"""
        self.shield_buffer.pop(0)
        self.shield_buffer.append(current_shield_up)
        new_state = "Up" if sum(self.shield_buffer) > len(self.shield_buffer) / 2 else "Down"
        
        # Perform shield click if state changed
        if new_state != self.shield_state:
            self.perform_shield_click(new_state == "Up")
            self.last_shield_state = self.shield_state
        
        self.shield_state = new_state
        return self.shield_state

    # ... (rest of ShieldDetector methods remain the same)

class GameDisplay:
    """Handles the display and visualization of game state"""
    @staticmethod
    def draw_status(frame: np.ndarray, swing_detector: SwingDetector, shield_state: str):
        """Draw game status information on the frame"""
        # Basic status
        cv2.putText(frame, f"State: {swing_detector.swing_state}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Total Swings: {swing_detector.total_swings}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Swing detection
        if swing_detector.cooldown_counter == swing_detector.config.COOLDOWN_FRAMES:
            cv2.putText(frame, "SWING!", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Shield status
        shield_color = (0, 255, 0) if shield_state == "Up" else (0, 0, 255)
        cv2.putText(frame, f"Shield: {shield_state}", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, shield_color, 2)
        
        # Block notification
        if shield_state == "Up":
            cv2.putText(frame, "BLOCKED!", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

def main():
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
    
    # Initialize detectors
    config = GameConfig()
    swing_detector = SwingDetector(config)
    shield_detector = ShieldDetector(config)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use built-in webcam
    cap.set(cv2.CAP_PROP_FPS, config.FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands for shield detection first
        results_hands = hands.process(rgb_frame)
        current_shield_up = False
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, 
                                                results_hands.multi_handedness):
                if shield_detector.is_left_hand(handedness):
                    current_shield_up = shield_detector.is_stop_gesture(hand_landmarks.landmark)
                    color = (0, 255, 0) if current_shield_up else (255, 0, 0)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=color, thickness=2),
                                           mp_drawing.DrawingSpec(color=color, thickness=2))
        
        shield_state = shield_detector.update_shield_state(current_shield_up)
        
        # Process pose for attack detection
        results_pose = pose.process(rgb_frame)
        swing_detected = False
        if results_pose.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results_pose.pose_landmarks.landmark]
            swing_detected = swing_detector.detect_swing(landmarks, shield_state)  # Pass shield state
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        GameDisplay.draw_status(frame, swing_detector, shield_state)
        
        cv2.imshow("Attack and Defend Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()