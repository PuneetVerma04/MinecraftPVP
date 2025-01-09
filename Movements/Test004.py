import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class IPWebcamMovementDetector:
    def __init__(self, webcam_url, scaling_factor=0.5):
        # Initialize MediaPipe components for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Video capture settings
        self.webcam_url = webcam_url
        self.scaling_factor = scaling_factor
        self.cap = None
        
        # Movement detection parameters
        self.MOVEMENT_THRESHOLD = 30  # Sensitivity of movement detection
        self.SMOOTHING_WINDOW = 5     # Number of frames to smooth movement over
        self.position_history = deque(maxlen=self.SMOOTHING_WINDOW)
        self.last_movement = None
        self.movement_cooldown = 0
    
    def initialize_capture(self):
        """Initialize connection to IP webcam"""
        self.cap = cv2.VideoCapture(self.webcam_url)
        if not self.cap.isOpened():
            raise ConnectionError("Failed to connect to IP webcam")
        return True

    def calculate_body_reference(self, landmarks):
        """
        Calculate a stable reference point by combining multiple body landmarks.
        Uses a weighted average of hip and shoulder positions for better stability.
        """
        # Calculate midpoint of hips
        hip_mid = np.array([
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x) / 2,
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
        ])
        
        # Calculate midpoint of shoulders
        shoulder_mid = np.array([
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2,
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        ])
        
        # Return weighted average (60% hips, 40% shoulders)
        return 0.6 * hip_mid + 0.4 * shoulder_mid

    def process_frame(self, frame):
        """Process a single frame for movement detection"""
        # Resize frame according to scaling factor
        new_width = int(frame.shape[1] * self.scaling_factor)
        new_height = int(frame.shape[0] * self.scaling_factor)
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)
        
        movement = None
        
        if result.pose_landmarks:
            # Calculate current position using body references
            current_pos = self.calculate_body_reference(result.pose_landmarks.landmark)
            
            # Convert normalized coordinates to pixel values
            current_pos = np.array([
                current_pos[0] * new_width,
                current_pos[1] * new_height
            ])
            
            # Add position to history for movement smoothing
            self.position_history.append(current_pos)
            
            # Detect movement when we have enough position history
            if len(self.position_history) == self.SMOOTHING_WINDOW:
                # Calculate movement vector from oldest to newest position
                movement_vector = current_pos - self.position_history[0]
                
                # Handle movement cooldown
                if self.movement_cooldown > 0:
                    self.movement_cooldown -= 1
                
                # Detect new movement if cooldown is complete
                if self.movement_cooldown == 0:
                    # Determine primary movement direction
                    if abs(movement_vector[1]) > abs(movement_vector[0]):
                        if movement_vector[1] < -self.MOVEMENT_THRESHOLD:
                            movement = "Forward"
                        elif movement_vector[1] > self.MOVEMENT_THRESHOLD:
                            movement = "Backward"
                    else:
                        if movement_vector[0] < -self.MOVEMENT_THRESHOLD:
                            movement = "Left"
                        elif movement_vector[0] > self.MOVEMENT_THRESHOLD:
                            movement = "Right"
                    
                    # Set cooldown if movement detected
                    if movement:
                        self.last_movement = movement
                        self.movement_cooldown = 10
            
            # Draw pose landmarks and connections
            self.mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Display movement direction on frame
            if self.last_movement:
                cv2.putText(frame, f"Movement: {self.last_movement}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
        
        return frame, movement

    def run(self):
        """Main loop for movement detection"""
        try:
            self.initialize_capture()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame and detect movement
                processed_frame, movement = self.process_frame(frame)
                if movement:
                    print(f"Detected movement: {movement}")
                
                # Display the processed frame
                cv2.imshow("IP Webcam Movement Detection", processed_frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Clean up resources
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

def main():
    # Replace with your IP Webcam URL
    webcam_url = "http://192.168.137.139:8080/video"
    
    # Create and run the detector
    detector = IPWebcamMovementDetector(webcam_url, scaling_factor=0.5)
    detector.run()

if __name__ == "__main__":
    main()