import cv2
import mediapipe as mp
import numpy as np

class DistanceDetector:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close range, 1 for long range
            min_detection_confidence=0.5
        )
        
        # Constants for distance calculation
        self.KNOWN_DISTANCE = 60  # calibration distance in cm
        self.KNOWN_FACE_WIDTH = 16  # average face width in cm
        
        # Calibrate the system with a face at known distance
        self.CALIBRATED_FACE_PIXELS = None
        
    def calibrate(self, face_width_pixels):
        """Store the face width in pixels at calibration distance"""
        self.CALIBRATED_FACE_PIXELS = face_width_pixels
        
    def calculate_distance(self, face_width_pixels):
        """Calculate distance using triangle similarity"""
        if self.CALIBRATED_FACE_PIXELS is None:
            self.calibrate(face_width_pixels)
            return self.KNOWN_DISTANCE
            
        distance = (self.KNOWN_FACE_WIDTH * self.CALIBRATED_FACE_PIXELS) / face_width_pixels
        return distance
        
    def process_frame(self, frame):
        """Process a single frame and return distance estimation"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect faces
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            # Get the first detected face
            detection = results.detections[0]
            
            # Get bounding box coordinates
            bbox = detection.location_data.relative_bounding_box
            
            # Get frame dimensions
            h, w, _ = frame.shape
            
            # Calculate face width in pixels
            face_width_pixels = bbox.width * w
            
            # Calculate distance
            distance = self.calculate_distance(face_width_pixels)
            
            # Draw bounding box
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Display distance
            distance_text = f"Distance: {distance:.1f} cm"
            cv2.putText(frame, distance_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return frame, distance
        
        return frame, None

def main():
    # Initialize the detector
    detector = DistanceDetector()
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame, distance = detector.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Distance Detection', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()