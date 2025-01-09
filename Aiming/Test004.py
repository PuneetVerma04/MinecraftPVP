import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# Disable PyAutoGUI's failsafe for smoother operation
pyautogui.FAILSAFE = False

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Create a shorter smoothing buffer for more responsive movement
position_history = deque(maxlen=3)  # Reduced from 5 to 3 for faster response

# Define key facial landmarks for tracking
NOSE_TIP = 4
LEFT_EYE = 133
RIGHT_EYE = 362
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Enhanced sensitivity settings
SENSITIVITY = {
    'x': 2.7,  # Increased from 1.5 to 2.5
    'y': 2.3,  # Increased from 1.5 to 2.2
    'acceleration': 1.3  # New acceleration factor
}

def calculate_face_center(landmarks):
    """Calculate the center point between eyes and nose tip for stable tracking"""
    nose = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
    left_eye = np.array([landmarks[LEFT_EYE].x, landmarks[LEFT_EYE].y])
    right_eye = np.array([landmarks[RIGHT_EYE].x, landmarks[RIGHT_EYE].y])
    
    # Adjusted weights to favor nose position more for precise control
    center = nose * 0.7 + (left_eye + right_eye) * 0.15  # Changed from 0.6/0.2 to 0.7/0.15
    return center

def smooth_movement(current_pos):
    """Apply smoothing with reduced lag"""
    position_history.append(current_pos)
    if len(position_history) < 2:
        return current_pos
    
    # Adjusted weights to favor current position more
    weights = np.array([0.6, 0.3, 0.1][:len(position_history)])  # Changed weights for more immediate response
    weights = weights / weights.sum()
    
    positions = np.array(position_history)
    smoothed = np.average(positions, weights=weights, axis=0)
    return smoothed.astype(int)

def map_to_screen(point, frame_width, frame_height):
    """Map face position to screen coordinates with enhanced sensitivity and acceleration"""
    # Calculate movement from center
    x_offset = point[0] - 0.5
    y_offset = point[1] - 0.5
    
    # Apply non-linear acceleration for more precise control
    x_offset = np.sign(x_offset) * (abs(x_offset) ** SENSITIVITY['acceleration'])
    y_offset = np.sign(y_offset) * (abs(y_offset) ** SENSITIVITY['acceleration'])
    
    # Apply sensitivity scaling
    x = 0.5 + (x_offset * SENSITIVITY['x'])
    y = 0.5 + (y_offset * SENSITIVITY['y'])
    
    # Map to screen coordinates
    screen_x = int(np.clip(x * screen_width, 0, screen_width))
    screen_y = int(np.clip(y * screen_height, 0, screen_height))
    
    return screen_x, screen_y

# Initialize video capture with optimal settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Performance monitoring
frame_times = deque(maxlen=30)

# Initialize calibration variables
calibration_center = None
calibration_frames = 0
CALIBRATION_DURATION = 20  # Reduced from 30 to 20 frames for faster startup

print("Hold your head steady in your preferred center position for calibration...")
print("Controls:")
print("- Press '+' to increase sensitivity")
print("- Press '-' to decrease sensitivity")
print("- Press 'Esc' to exit")

try:
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for more intuitive movement
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB and process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate face center position
            face_center = calculate_face_center(face_landmarks.landmark)
            
            # Handle calibration
            if calibration_frames < CALIBRATION_DURATION:
                if calibration_center is None:
                    calibration_center = face_center
                else:
                    calibration_center = calibration_center * 0.8 + face_center * 0.2  # Faster calibration
                calibration_frames += 1
                
                # Draw calibration progress
                progress = int((calibration_frames / CALIBRATION_DURATION) * 100)
                cv2.putText(frame, f"Calibrating: {progress}%", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Map to screen coordinates with enhanced sensitivity
                screen_pos = map_to_screen(face_center, frame.shape[1], frame.shape[0])
                
                # Apply smoothing
                smoothed_pos = smooth_movement(np.array(screen_pos))
                
                # Move mouse cursor
                pyautogui.moveTo(*smoothed_pos, _pause=False)
                
                # Draw crosshair and sensitivity info
                center_x = int(face_center[0] * frame.shape[1])
                center_y = int(face_center[1] * frame.shape[0])
                cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0), 
                             cv2.MARKER_CROSS, 20, 2)
                
                # Display current sensitivity
                cv2.putText(frame, f"Sensitivity: {SENSITIVITY['x']:.1f}x", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Handle keyboard input for sensitivity adjustment
        key = cv2.waitKey(1) & 0xFF
        if key == ord('+') or key == ord('='):
            SENSITIVITY['x'] += 0.1
            SENSITIVITY['y'] += 0.1
        elif key == ord('-') or key == ord('_'):
            SENSITIVITY['x'] = max(1.0, SENSITIVITY['x'] - 0.1)
            SENSITIVITY['y'] = max(1.0, SENSITIVITY['y'] - 0.1)
        elif key == 27:  # Esc
            break
        
        # Calculate and display FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        fps = 1 / (sum(frame_times) / len(frame_times))
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Tracking', frame)

finally:
    cap.release()
    cv2.destroyAllWindows()