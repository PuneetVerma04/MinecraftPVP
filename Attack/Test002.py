import cv2
import mediapipe as mp
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class GestureConfig:
    """Configuration settings for gesture detection"""
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    movement_threshold: float = 0.15  # seconds
    velocity_threshold: float = 1000  # pixels per second
    frame_width: int = 1920
    frame_height: int = 1080
    display_width: int = 1280  # Display window width
    display_height: int = 720   # Display window height
    upper_line_y: int = 150    # Adjusted for new resolution
    lower_line_y: int = 600    # Adjusted for new resolution
    hand_to_track: str = "Right"  # "Right", "Left", or "Both"
    fist_threshold: float = 0.1  # threshold for detecting closed fist

class HandGestureDetector:
    def __init__(self, config: GestureConfig):
        """Initialize the hand gesture detector with given configuration"""
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.prev_y = None
        self.prev_time = None
        self.attack_time = None
        self.crossed_upper = False
        self.crossed_lower = False
        
    def setup_camera(self) -> Optional[cv2.VideoCapture]:
        """Initialize and configure the camera"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        return cap
    
    def resize_frame(self, frame: cv2.Mat) -> cv2.Mat:
        """Resize frame to display resolution while maintaining aspect ratio"""
        aspect_ratio = self.config.frame_width / self.config.frame_height
        target_aspect = self.config.display_width / self.config.display_height
        
        if aspect_ratio > target_aspect:
            # Width limited
            new_width = self.config.display_width
            new_height = int(new_width / aspect_ratio)
        else:
            # Height limited
            new_height = self.config.display_height
            new_width = int(new_height * aspect_ratio)
            
        return cv2.resize(frame, (new_width, new_height))

    def is_fist_closed(self, hand_landmarks) -> bool:
        """
        Detect if the hand is in a fist position by checking if fingers are curled
        """
        fingertips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_bases = [
            self.mp_hands.HandLandmark.THUMB_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]
        
        is_closed = True
        for tip, base in zip(fingertips, finger_bases):
            tip_y = hand_landmarks.landmark[tip].y
            base_y = hand_landmarks.landmark[base].y
            
            if tip == self.mp_hands.HandLandmark.THUMB_TIP:
                palm_center = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y
                if abs(tip_y - palm_center) > self.config.fist_threshold:
                    is_closed = False
                    break
            elif tip_y < base_y:
                is_closed = False
                break
                
        return is_closed

    def calculate_velocity(self, current_y: float, current_time: float) -> Optional[float]:
        """Calculate the vertical velocity of hand movement"""
        if self.prev_y is None or self.prev_time is None:
            return None
            
        dy = abs(current_y - self.prev_y)
        dt = current_time - self.prev_time
        return dy / dt if dt > 0 else 0

    def process_frame(self, frame: cv2.Mat) -> Tuple[cv2.Mat, bool]:
        """Process a single frame and return the processed frame and attack detection status"""
        frame = cv2.flip(frame, 1)
        frame_height = frame.shape[0]
        
        # Scale line positions based on frame height
        upper_line_y = int(self.config.upper_line_y * frame_height / self.config.display_height)
        lower_line_y = int(self.config.lower_line_y * frame_height / self.config.display_height)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        attack_detected = False
        
        # Draw reference lines
        cv2.line(frame, (0, upper_line_y), (frame.shape[1], upper_line_y), (0, 255, 0), 2)
        cv2.line(frame, (0, lower_line_y), (frame.shape[1], lower_line_y), (0, 255, 0), 2)
        
        # Process hands
        result = self.hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            attack_detected = self._process_hand_landmarks(frame, result, upper_line_y, lower_line_y)
        else:
            # Reset crossing flags when no hand is detected
            self.crossed_upper = False
            self.crossed_lower = False
            
        # Resize frame for display
        frame = self.resize_frame(frame)
        return frame, attack_detected
    
    def _process_hand_landmarks(self, frame: cv2.Mat, result, upper_line_y: int, lower_line_y: int) -> bool:
        """Process hand landmarks and detect attacks"""
        attack_detected = False
        current_time = time.time()
        
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            handedness = result.multi_handedness[idx].classification[0].label
            
            if self.config.hand_to_track != "Both" and handedness != self.config.hand_to_track:
                continue
                
            # Get wrist position
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            wrist_y = int(wrist.y * frame.shape[0])
            
            # Check if hand is in fist position
            is_fist = self.is_fist_closed(hand_landmarks)
            
            if is_fist:
                # Check line crossing
                if wrist_y <= upper_line_y and not self.crossed_upper:
                    self.crossed_upper = True
                    self.crossed_lower = False
                elif wrist_y >= lower_line_y and not self.crossed_lower:
                    self.crossed_lower = True
                    self.crossed_upper = False
                
                # Detect attack pattern
                if (self.crossed_upper and wrist_y >= lower_line_y) or (self.crossed_lower and wrist_y <= upper_line_y):
                    current_time = time.time()
                    if self.attack_time is None or (current_time - self.attack_time > self.config.movement_threshold):
                        self.attack_time = current_time
                        attack_detected = True
                        # Display "ATTACK!" in large red text
                        font_scale = frame.shape[0] / 500.0  # Scale font based on frame size
                        cv2.putText(frame, "ATTACK!", 
                                  (int(frame.shape[1]/3), int(frame.shape[0]/2)),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 3)
                    self.crossed_upper = False
                    self.crossed_lower = False
            
            # Draw hand state
            hand_state = "Fist" if is_fist else "Open"
            cv2.putText(frame, f"Hand: {hand_state}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw landmarks
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
        return attack_detected
    
    def run(self):
        """Main loop for the hand gesture detector"""
        cap = self.setup_camera()
        if cap is None:
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                processed_frame, attack_detected = self.process_frame(frame)
                if attack_detected:
                    print("Attack detected!")
                
                cv2.imshow("Attack Detection", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main entry point"""
    config = GestureConfig()
    detector = HandGestureDetector(config)
    detector.run()

if __name__ == "__main__":
    main()