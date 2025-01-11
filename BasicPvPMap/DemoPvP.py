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
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key

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
    MOVEMENT_VERTICAL_THRESHOLD: int = 5
    MOVEMENT_HORIZONTAL_THRESHOLD: int = 30
    MOVEMENT_SMOOTHING_WINDOW: int = 5

@dataclass
class SensitivitySettings:
    """Configuration for head tracking sensitivity"""
    x: float = 8.0
    y: float = 8.0
    acceleration: float = 1.1
    dead_zone: float = 0.03
    movement_scale: float = 30.0
    max_movement: float = 50.0

class CombatSystem:
    """Handles attack and defense detection"""
    def __init__(self, config: GameConfig):
        self.config = config
        self.landmark_history = deque(maxlen=config.HISTORY_SIZE)
        self.swing_state = "ready"
        self.state_counter = 0
        self.total_swings = 0
        self.cooldown_counter = 0
        self.last_click_state = False
        self.shield_state = "Down"
        self.last_shield_state = "Down"
        
    def perform_attack_click(self):
        """Perform left mouse click for attack"""
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self.last_click_state = True

    def perform_shield_click(self, shield_up: bool):
        """Perform right mouse click for shield"""
        if shield_up and self.last_shield_state == "Down":
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        elif not shield_up and self.last_shield_state == "Up":
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def get_wrist_position(self, landmarks: List[List[float]]) -> float:
        """Calculate relative position of right wrist to right shoulder"""
        if not landmarks:
            return 0
        right_shoulder = np.array([landmarks[12][1]])
        right_wrist = np.array([landmarks[16][1]])
        return float(right_shoulder - right_wrist)

    def detect_swing(self, landmarks: List[List[float]], shield_state: str) -> bool:
        """Detect sword swing motion"""
        self.landmark_history.append(landmarks)
        
        if shield_state == "Up":
            self.swing_state = "ready"
            self.state_counter = 0
            self.last_click_state = False
            return False
        
        if len(self.landmark_history) < 2 or self.cooldown_counter > 0:
            self.cooldown_counter = max(0, self.cooldown_counter - 1)
            return False
            
        current_pos = self.get_wrist_position(self.landmark_history[-1])
        prev_pos = self.get_wrist_position(self.landmark_history[-2])
        pos_change = current_pos - prev_pos
        
        FAST_MOVEMENT_THRESHOLD = 0.05
        SWING_DETECTION_THRESHOLD = 1
        
        swing_detected = False
        
        if self.swing_state == "ready":
            self.last_click_state = False
            if pos_change < -FAST_MOVEMENT_THRESHOLD:
                self.swing_state = "swinging"
                self.state_counter = 0
                
        elif self.swing_state == "swinging":
            if pos_change < -FAST_MOVEMENT_THRESHOLD:
                self.state_counter += 1
                if self.state_counter >= SWING_DETECTION_THRESHOLD:
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

    def is_stop_gesture(self, landmarks) -> bool:
        """Detect stop gesture for shield"""
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

    def is_both_hands_raised(self, hand_landmarks, handedness) -> bool:
        """Check if both hands are raised in a stop gesture Returns True if both hands are detected and raised """
        left_hand_raised = False
        right_hand_raised = False
    
        for landmarks, hand in zip(hand_landmarks, handedness):
            # Check hand type (Left appears as "Left" in mirrored view)
            is_left = hand.classification[0].label == "Left"
        
            # Similar to stop gesture check but simplified for both hands
            wrist = landmarks.landmark[0]
            fingertips = [landmarks.landmark[tip] for tip in [8, 12, 16, 20]]  # Index to pinky tips
        
             # Check if hand is raised (all fingertips above wrist)
            hand_raised = all(tip.y < wrist.y for tip in fingertips)
        
            if is_left:
                left_hand_raised = hand_raised
            else:
                right_hand_raised = hand_raised
            
        return left_hand_raised and right_hand_raised

    def update_shield_state(self, current_shield_up: bool) -> str:
        """Update shield state"""
        new_state = "Up" if current_shield_up else "Down"
        
        if new_state != self.shield_state:
            self.perform_shield_click(new_state == "Up")

        self.last_shield_state = self.shield_state
        self.shield_state = new_state
        return self.shield_state

class MovementSystem:
    """Handles movement detection and WASD control"""
    def __init__(self, config: GameConfig):
        self.config = config
        self.position_history = deque(maxlen=config.MOVEMENT_SMOOTHING_WINDOW)
        self.last_movement = None
        self.movement_cooldown = 0
        self.state = None
        self.keyboard = KeyboardController()
        self.current_actions = {}

    def calculate_body_reference(self, landmarks) -> np.ndarray:
        """Calculate stable body reference point"""
        hip_mid = np.array([
            (landmarks[23].x + landmarks[24].x) / 2,
            (landmarks[23].y + landmarks[24].y) / 2
        ])
        
        shoulder_mid = np.array([
            (landmarks[11].x + landmarks[12].x) / 2,
            (landmarks[11].y + landmarks[12].y) / 2
        ])
        
        return 0.6 * hip_mid + 0.4 * shoulder_mid

    def detect_movement(self, frame_width: int, frame_height: int, landmarks) -> Optional[int]:
        """Detect movement direction"""
        if not landmarks:
            return None

        current_pos = self.calculate_body_reference(landmarks)
        current_pos = np.array([
            current_pos[0] * frame_width,
            current_pos[1] * frame_height
        ])
        
        self.position_history.append(current_pos)
        
        if len(self.position_history) == self.config.MOVEMENT_SMOOTHING_WINDOW:
            if self.movement_cooldown > 0:
                self.movement_cooldown -= 1
                return self.state
            
            movement_vector = current_pos - self.position_history[0]
            abs_x = abs(movement_vector[0])
            abs_y = abs(movement_vector[1])
            
            vertical_significant = abs_y > self.config.MOVEMENT_VERTICAL_THRESHOLD
            horizontal_significant = abs_x > self.config.MOVEMENT_HORIZONTAL_THRESHOLD
            
            if vertical_significant and (not horizontal_significant or abs_y > abs_x):
                self.state = 0 if movement_vector[1] < 0 else 1  # Forward/Backward
            elif horizontal_significant:
                self.state = 3 if movement_vector[0] < 0 else 2  # Left/Right
            
            self.movement_cooldown = 10

        return self.state

    def map_movement(self, state):
        """Map movement state to keyboard controls"""
        if state == 0:  # Forward
            if self.current_actions.get("Forward") != 0:
                self.keyboard.press('w')
                self.keyboard.press(Key.ctrl_l)
                self.current_actions["Forward"] = 0
        else:
            if self.current_actions.get("Forward") == 0:
                self.keyboard.release('w')
                self.keyboard.release(Key.ctrl_l)
                self.current_actions["Forward"] = None

        movement_keys = {1: 's', 2: 'a', 3: 'd'}
        movement_names = {1: "Backward", 2: "Left", 3: "Right"}
        
        if state in movement_keys:
            if self.current_actions.get(movement_names[state]) != state:
                self.keyboard.press(movement_keys[state])
                self.current_actions[movement_names[state]] = state

    def stop_movement(self, state):
        """Stop movement when state changes"""
        movement_map = {
            0: ('w', 'Forward', Key.ctrl_l),
            1: ('s', 'Backward', None),
            2: ('a', 'Left', None),
            3: ('d', 'Right', None)
        }
        
        if state in movement_map:
            key, action_name, extra_key = movement_map[state]
            if self.current_actions.get(action_name) == state:
                self.keyboard.release(key)
                if extra_key:
                    self.keyboard.release(extra_key)
                self.current_actions[action_name] = None
    def stop_all_movements(self):
        """Stop all ongoing movements"""
        movement_keys = ['w', 'a', 's', 'd']
        for key in movement_keys:
            self.keyboard.release(key)
        self.keyboard.release(Key.ctrl_l)
        self.current_actions.clear()
        self.last_movement = None
        self.state = None
    
    def cleanup(self):
        """Release all keys"""
        for action in self.current_actions.keys():
            self.stop_movement(action)

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

class GameController:
    """Main game controller handling combat, movement, and head tracking"""
    def __init__(self):
        self.config = GameConfig()
        self.combat_system = CombatSystem(self.config)
        self.movement_system = MovementSystem(self.config)
        self.head_tracker = HeadTracker()
        
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
        
    def process_frame(self, frame):
        """Process a single frame for combat, movement, and head tracking"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        frame_height, frame_width = frame.shape[:2]

        # Process head tracking
        face_results = self.head_tracker.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            self.head_tracker.process_landmarks(face_results.multi_face_landmarks[0])

        # Process hands for shield detection
        results_hands = self.hands.process(rgb_frame)
        current_shield_up = False
        both_hands_raised = False

        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            # First check for both hands raised
            both_hands_raised = self.combat_system.is_both_hands_raised(
                results_hands.multi_hand_landmarks, 
                results_hands.multi_handedness
            )
    
            if both_hands_raised:
                self.emergency_stop()
                # Draw both hands in yellow to indicate emergency stop
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
            else:
                # Normal shield detection for single hand
                for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, 
                                            results_hands.multi_handedness):
                    if handedness.classification[0].label == "Right":  # Left hand in mirror
                        current_shield_up = self.combat_system.is_stop_gesture(hand_landmarks.landmark)
                        color = (0, 255, 0) if current_shield_up else (255, 0, 0)
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=color, thickness=2),
                            self.mp_drawing.DrawingSpec(color=color, thickness=2)
                        )

        # Only process shield state if both hands are not raised
        if not both_hands_raised:
            shield_state = self.combat_system.update_shield_state(current_shield_up)
        else:
            shield_state = "Down"

        # Process pose for movement and attack
        results_pose = self.pose.process(rgb_frame)
        if results_pose.pose_landmarks:
            # Handle movement detection
            movement_state = self.movement_system.detect_movement(
                frame_width, frame_height, results_pose.pose_landmarks.landmark)
        
            if movement_state is not None:
                self.movement_system.map_movement(movement_state)
                if movement_state != self.movement_system.last_movement and self.movement_system.last_movement is not None:
                    self.movement_system.stop_movement(self.movement_system.last_movement)
                self.movement_system.last_movement = movement_state

            # Handle attack detection
            landmarks = [[lm.x, lm.y, lm.z] for lm in results_pose.pose_landmarks.landmark]
            swing_detected = self.combat_system.detect_swing(landmarks, shield_state)
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Display status information
            self.draw_status(frame, movement_state, shield_state, swing_detected)

        return frame

    def draw_status(self, frame, movement_state, shield_state, swing_detected):
        """Draw status information on the frame"""
        # Shield status
        shield_color = (0, 255, 0) if shield_state == "Up" else (0, 0, 255)
        cv2.putText(frame, f"Shield: {shield_state}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, shield_color, 2)

        # Movement status
        movement_names = {0: "Forward", 1: "Backward", 2: "Left", 3: "Right"}
        movement_text = movement_names.get(movement_state, "None")
        cv2.putText(frame, f"Movement: {movement_text}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Combat status
        attack_text = "Swinging" if swing_detected else "Ready"
        cv2.putText(frame, f"Attack: {attack_text}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Total swings
        cv2.putText(frame, f"Total Swings: {self.combat_system.total_swings}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def emergency_stop(self):
        """Stop all game actions when both hands are raised"""
        self.movement_system.stop_all_movements()
        if self.combat_system.shield_state == "Up":
            self.combat_system.perform_shield_click(False)
            self.combat_system.shield_state = "Down"
            self.combat_system.last_shield_state = "Down"

    def cleanup(self):
        """Clean up resources and release all keys"""
        self.movement_system.cleanup()
        self.head_tracker.face_mesh.close()
        self.pose.close()
        self.hands.close()

def main():
    """Main execution function"""
    print("\nUnified Game Control System:")
    print("Press 'R' to recenter camera view")
    print("Press 'Q' to exit")
    print("\nControls:")
    print("- Head movement: Look around")
    print("- Body movement: WASD controls")
    print("- Right hand swing: Attack")
    print("- Left hand stop gesture: Shield\n")

    controller = GameController()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, controller.config.FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = controller.process_frame(frame)
            cv2.imshow("Unified Game Control System", frame)
            
            # Handle keyboard input
            if keyboard.is_pressed('r'):
                controller.head_tracker.is_calibrating = True
                print("Recentering camera view...")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        controller.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()