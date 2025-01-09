import cv2
import mediapipe as mp
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Deque
from collections import deque

@dataclass
class GestureConfig:
    """Optimized configuration settings for low-latency detection"""
    # Lower resolution for faster processing
    frame_width: int = 640
    frame_height: int = 480
    display_width: int = 640
    display_height: int = 480
    
    # Reduced confidence thresholds for faster detection
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Optimized timing parameters
    movement_threshold: float = 0.08    # Faster movement detection
    velocity_threshold: float = 800     # Adjusted for lower resolution
    
    # Gesture detection zones (adjusted for lower resolution)
    upper_line_y: int = 140
    lower_line_y: int = 340
    
    # Performance optimization settings
    max_hands: int = 1                  # Track only one hand for better performance
    hand_to_track: str = "Right"        # Focus on single hand
    buffer_size: int = 3                # Smaller buffer for lower latency
    max_frame_drop: int = 2             # Reduced for faster recovery
    fist_threshold: float = 0.12        # Balanced threshold

class HandGestureDetector:
    def __init__(self, config: GestureConfig):
        """Initialize detector with performance optimizations"""
        self.config = config
        
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,          # Optimize for video
            max_num_hands=config.max_hands,   # Limit hand detection
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            model_complexity=0                # Use fastest model
        )
        
        # Simplified drawing utilities
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)
        
        # Efficient circular buffers for position tracking
        self.position_history = deque(maxlen=config.buffer_size)
        self.time_history = deque(maxlen=config.buffer_size)
        
        # State tracking variables
        self.crossed_upper = False
        self.crossed_lower = False
        self.last_attack_time = 0
        self.frames_since_detection = 0

    def setup_camera(self) -> Optional[cv2.VideoCapture]:
        """Set up camera with optimized settings"""
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows for better performance
            if not cap.isOpened():
                return None
                
            # Optimize camera settings for performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            cap.set(cv2.CAP_PROP_FPS, 60)  # Request higher FPS
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay
            
            return cap
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return None

    def is_fist_closed(self, hand_landmarks) -> bool:
        """Optimized fist detection focusing on key landmarks"""
        # Check only thumb, index, and middle finger for faster processing
        key_points = [
            (self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.THUMB_MCP),
            (self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_MCP),
            (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
        ]
        
        curl_sum = 0
        for tip, base in key_points:
            tip_y = hand_landmarks.landmark[tip].y
            base_y = hand_landmarks.landmark[base].y
            curl_sum += tip_y - base_y
            
        return (curl_sum / len(key_points)) > self.config.fist_threshold

    def process_frame(self, frame: cv2.Mat) -> Tuple[cv2.Mat, bool]:
        """Process frame with optimized detection pipeline"""
        # Flip only if needed for display
        frame = cv2.flip(frame, 1)
        frame_height = frame.shape[0]
        
        # Calculate lines once
        upper_line_y = self.config.upper_line_y
        lower_line_y = self.config.lower_line_y
        
        # Process image in RGB without copying
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Prevent copying in MediaPipe
        result = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        # Draw reference lines (minimal drawing)
        cv2.line(frame, (0, upper_line_y), (frame.shape[1], upper_line_y), (0, 255, 0), 1)
        cv2.line(frame, (0, lower_line_y), (frame.shape[1], lower_line_y), (0, 255, 0), 1)
        
        attack_detected = False
        current_time = time.time()
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Process only the first detected hand
                wrist_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame_height)
                
                # Update position tracking
                self.position_history.append(wrist_y)
                self.time_history.append(current_time)
                
                if self.is_fist_closed(hand_landmarks):
                    attack_detected = self._check_attack_pattern(wrist_y, upper_line_y, lower_line_y, current_time)
                    
                    if attack_detected:
                        # Draw attack indicator efficiently
                        cv2.putText(frame, "ATTACK!", (frame.shape[1]//3, frame.shape[0]//2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Minimal landmark drawing
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_spec, self.drawing_spec
                )
                break  # Process only first hand
                
        return frame, attack_detected

    def _check_attack_pattern(self, y: float, upper_line_y: int, lower_line_y: int, current_time: float) -> bool:
        """Optimized attack pattern detection"""
        # Update crossing flags
        if y <= upper_line_y and not self.crossed_upper:
            self.crossed_upper = True
            self.crossed_lower = False
        elif y >= lower_line_y and not self.crossed_lower:
            self.crossed_lower = True
            self.crossed_upper = False
            
        # Check for attack pattern
        if ((self.crossed_upper and y >= lower_line_y) or 
            (self.crossed_lower and y <= upper_line_y)):
            if current_time - self.last_attack_time > self.config.movement_threshold:
                self.last_attack_time = current_time
                self.crossed_upper = False
                self.crossed_lower = False
                return True
        return False

    def run(self):
        """Main loop with performance optimizations"""
        cap = self.setup_camera()
        if cap is None:
            return
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frame, attack_detected = self.process_frame(frame)
                
                if attack_detected:
                    print("Attack!")
                    
                cv2.imshow("Attack Detection", processed_frame)
                
                # Efficient key checking
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Entry point with error handling"""
    try:
        config = GestureConfig()
        detector = HandGestureDetector(config)
        detector.run()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()