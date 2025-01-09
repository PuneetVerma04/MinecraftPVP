import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time
from collections import deque

# Disable pyautogui's failsafe for better performance
pyautogui.FAILSAFE = False

# Initialize MediaPipe with performance optimizations
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Optimize for video
    max_num_hands=1,          # Track only one hand
    min_detection_confidence=0.5,  # Lower threshold for faster detection
    min_tracking_confidence=0.5,   # Lower threshold for faster tracking
    model_complexity=0        # Use fastest model (0=Lite, 1=Full)
)

# Get screen dimensions once at startup
screen_width, screen_height = pyautogui.size()

# Pre-allocate numpy arrays for frame processing
frame_shape = None

# Initialize coordinate history for advanced smoothing
coord_history = deque(maxlen=3)  # Shorter history for lower latency

def calculate_distance(point1, point2):
    """Optimized distance calculation using numpy"""
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def optimize_frame(frame):
    """Optimize frame for processing"""
    # Resize frame to reduce processing time
    return cv2.resize(frame, (640, 480))

def smooth_coords(current_coords):
    """Enhanced smoothing with shorter history and weighted average"""
    coord_history.append(current_coords)
    if len(coord_history) < 2:
        return current_coords
    
    # Weighted average favoring recent positions
    weights = np.array([0.7, 0.2, 0.1][:len(coord_history)])
    weights = weights / weights.sum()
    
    smoothed = np.average(list(coord_history), weights=weights, axis=0)
    return smoothed.astype(int)

# Initialize video capture with optimal settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduced resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced resolution
cap.set(cv2.CAP_PROP_FPS, 30)            # Optimal FPS
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimize buffer delay

# Performance monitoring
frame_times = deque(maxlen=30)

try:
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        # Optimize frame processing
        frame = optimize_frame(frame)
        frame = cv2.flip(frame, 1)
        
        # Process frame without copying
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Prevent copying
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand only
            
            # Fast array operations for landmark processing
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Optimized gesture detection
            threshold = 0.05
            is_fist = (calculate_distance(index_tip, index_mcp) < threshold and
                      calculate_distance(middle_tip, middle_mcp) < threshold)

            if is_fist:
                # Efficient coordinate calculation
                center_x = int((wrist.x + index_mcp.x + middle_mcp.x) / 3 * screen_width)
                center_y = int((wrist.y + index_mcp.y + middle_mcp.y) / 3 * screen_height)
                
                # Apply optimized smoothing
                smoothed_coords = smooth_coords(np.array([center_x, center_y]))
                pyautogui.moveTo(*smoothed_coords, _pause=False)  # Disable internal pauses

        # Calculate and display FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        fps = 1 / (sum(frame_times) / len(frame_times))
        
        # Display FPS (optional - remove for maximum performance)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()