import cv2
import mediapipe as mp
import numpy as np

class MovementDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.prev_x = None
        self.prev_y = None
        self.movement_threshold = 30  # Adjust this value to change sensitivity

    def detect_movement(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect pose
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Get nose landmark (point 0) for tracking overall body movement
            nose = results.pose_landmarks.landmark[0]
            frame_height, frame_width, _ = frame.shape
            
            # Convert coordinates to pixels
            current_x = int(nose.x * frame_width)
            current_y = int(nose.y * frame_height)
            
            # Initialize previous position if not set
            if self.prev_x is None:
                self.prev_x = current_x
                self.prev_y = current_y
                return "Stable"
            
            # Calculate movement
            x_diff = current_x - self.prev_x
            y_diff = current_y - self.prev_y
            
            # Determine direction based on movement
            direction = "Stable"
            if abs(x_diff) > self.movement_threshold or abs(y_diff) > self.movement_threshold:
                if abs(x_diff) > abs(y_diff):
                    direction = "Right" if x_diff > 0 else "Left"
                else:
                    direction = "Backward" if y_diff > 0 else "Forward"
            
            # Update previous position
            self.prev_x = current_x
            self.prev_y = current_y
            
            return direction
        
        return "No pose detected"

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    detector = MovementDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect movement
        movement = detector.detect_movement(frame)
        
        # Display direction on frame
        cv2.putText(frame, f"Movement: {movement}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Movement Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()