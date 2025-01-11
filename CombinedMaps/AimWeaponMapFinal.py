import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import win32api
import win32con
import keyboard

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

@dataclass
class SensitivitySettings:
    """Configuration for head tracking sensitivity"""
    x: float = 8.0
    y: float = 8.0
    acceleration: float = 1.1
    dead_zone: float = 0.03
    movement_scale: float = 30.0
    max_movement: float = 50.0

class SwingDetector:
    """Handles detection of sword swing motions using pose landmarks"""
    def __init__(self, config: GameConfig):
        self.config = config
        self.landmark_history = deque(maxlen=config.HISTORY_SIZE)
        self.swing_state = "ready"
        self.state_counter = 0
        self.total_swings = 0
        self.cooldown_counter = 0
        self.last_click_state = False
        # Adjusted thresholds for faster detection
        self.FAST_MOVEMENT_THRESHOLD = 0.05
        self.SWING_DETECTION_THRESHOLD = 1
        
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
        """Detect if a swing motion has occurred with the right hand"""
        self.landmark_history.append(landmarks)
        
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
        
        swing_detected = False
        
        if self.swing_state == "ready":
            self.reset_click_state()
            if pos_change < -self.FAST_MOVEMENT_THRESHOLD:
                self.swing_state = "swinging"
                self.state_counter = 0
                
        elif self.swing_state == "swinging":
            if pos_change < -self.FAST_MOVEMENT_THRESHOLD:
                self.state_counter += 1
                if self.state_counter >= self.SWING_DETECTION_THRESHOLD:
                    if not self.last_click_state:
                        self.perform_attack_click()
                    self.swing_state = "ready"
                    self.state_counter = 0
                    self.total_swings += 1
                    self.cooldown_counter = max(3, self.config.COOLDOWN_FRAMES // 2)
                    swing_detected = True
            else:
                self.swing_state = "ready"
                self.state_counter = 0
        
        return swing_detected

class ShieldDetector:
    """Handles detection of shield (stop gesture) using left hand landmarks"""
    def __init__(self, config: GameConfig):
        self.config = config
        self.shield_state = "Down"
        self.last_shield_state = "Down"

    @staticmethod
    def is_left_hand(handedness) -> bool:
        """Check if the detected hand is the left hand"""
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
        new_state = "Up" if current_shield_up else "Down"
        if new_state != self.shield_state:
            self.perform_shield_click(new_state == "Up")
        self.last_shield_state = self.shield_state
        self.shield_state = new_state
        return self.shield_state

class HeadTracker:
    """Handles head tracking for camera control"""
    FOREHEAD_TOP = 10
    CHIN_BOTTOM = 152

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
        self.calibration_center = None
        self.calibration_frames = 0
        self.CALIBRATION_DURATION = calibration_frames
        self.last_position = None
        self.is_calibrating = False

    def recenter_view(self, current_pos):
        """Recenter the view to current head position"""
        self.calibration_center = current_pos
        self.last_position = current_pos
        self.position_history.clear()

    def calculate_head_center(self, landmarks):
        """Calculate the center point between forehead and chin"""
        forehead = np.array([landmarks[self.FOREHEAD_TOP].x, landmarks[self.FOREHEAD_TOP].y])
        chin = np.array([landmarks[self.CHIN_BOTTOM].x, landmarks[self.CHIN_BOTTOM].y])
        return (forehead + chin) / 2

    def move_camera(self, dx: int, dy: int):
        """Apply camera movement based on head position"""
        dx = np.clip(dx, -self.sensitivity.max_movement, self.sensitivity.max_movement)
        dy = np.clip(dy, -self.sensitivity.max_movement, self.sensitivity.max_movement)
        if abs(dx) > 0 or abs(dy) > 0:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

    def calculate_relative_movement(self, current_pos):
        """Calculate relative movement from calibration center"""
        if self.last_position is None:
            self.last_position = current_pos
            return 0, 0

        x_offset = (current_pos[0] - self.calibration_center[0])
        y_offset = (current_pos[1] - self.calibration_center[1])

        if abs(x_offset) < self.sensitivity.dead_zone:
            x_offset = 0
        if abs(y_offset) < self.sensitivity.dead_zone:
            y_offset = 0

        x_movement = -np.sign(x_offset) * (abs(x_offset) ** self.sensitivity.acceleration)
        y_movement = np.sign(y_offset) * (abs(y_offset) ** self.sensitivity.acceleration)

        x_movement *= self.sensitivity.x * self.sensitivity.movement_scale
        y_movement *= self.sensitivity.y * self.sensitivity.movement_scale

        self.last_position = current_pos
        return x_movement, y_movement

    def process_landmarks(self, face_landmarks):
        """Process face landmarks for head tracking"""
        head_center = self.calculate_head_center(face_landmarks.landmark)

        if self.calibration_frames < self.CALIBRATION_DURATION or self.is_calibrating:
            if self.calibration_center is None or self.is_calibrating:
                self.calibration_center = head_center
                self.is_calibrating = False
            else:
                self.calibration_center = self.calibration_center * 0.8 + head_center * 0.2
            self.calibration_frames += 1
        else:
            dx, dy = self.calculate_relative_movement(head_center)
            if dx != 0 or dy != 0:
                self.move_camera(dx, dy)

class GameDisplay:
    """Handles the display and visualization of game state"""
    @staticmethod
    def draw_status(frame: np.ndarray, swing_detector: SwingDetector, shield_state: str):
        """Draw game status information on the frame"""
        cv2.putText(frame, f"State: {swing_detector.swing_state}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Total Swings: {swing_detector.total_swings}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if swing_detector.cooldown_counter == swing_detector.config.COOLDOWN_FRAMES:
            cv2.putText(frame, "SWING!", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        shield_color = (0, 255, 0) if shield_state == "Up" else (0, 0, 255)
        cv2.putText(frame, f"Shield: {shield_state}", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, shield_color, 2)
        
        if shield_state == "Up":
            cv2.putText(frame, "BLOCKED!", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

def main():
    print("\nGame Control System:")
    print("Press 'R' to recenter camera, 'Q' to exit")
    print("Use head movement for camera control")
    print("Right hand swing for attack")
    print("Left hand stop gesture for shield\n")

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize detectors
    config = GameConfig()
    swing_detector = SwingDetector(config)
    shield_detector = ShieldDetector(config)
    head_tracker = HeadTracker()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize pose and hands detection
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process head tracking
            face_results = head_tracker.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                head_tracker.process_landmarks(face_results.multi_face_landmarks[0])

            # Process hands for shield detection
            hands_results = hands.process(rgb_frame)
            current_shield_up = False
            if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
                for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, 
                                                    hands_results.multi_handedness):
                    if shield_detector.is_left_hand(handedness):
                        current_shield_up = shield_detector.is_stop_gesture(hand_landmarks.landmark)
                        color = (0, 255, 0) if current_shield_up else (255, 0, 0)
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=color, thickness=2),
                            mp_drawing.DrawingSpec(color=color, thickness=2)
                        )

            shield_state = shield_detector.update_shield_state(current_shield_up)
            pose_results = pose.process(rgb_frame)
            swing_detected = False
            if pose_results.pose_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]
                swing_detected = swing_detector.detect_swing(landmarks, shield_state)
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Reset frame writeable flag
            rgb_frame.flags.writeable = True

            # Draw status information
            GameDisplay.draw_status(frame, swing_detector, shield_state)

            # Handle keyboard input
            if keyboard.is_pressed('r'):
                if face_results.multi_face_landmarks:
                    head_tracker.is_calibrating = True
                print("Recentering camera view...")

            # Display frame
            cv2.imshow('Game Control System', frame)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        hands.close()
        head_tracker.face_mesh.close()

if __name__ == "__main__":
    main()