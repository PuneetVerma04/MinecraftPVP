import cv2
import mediapipe as mp
import pyautogui
import math
from collections import deque

# Initialization - disable pyautogui's failsafe for smoother operation
pyautogui.FAILSAFE = False

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils  # Add drawing utilities for visualization

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def smooth_coords(current, previous, alpha=0.8):
    """Apply exponential smoothing to coordinates"""
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous

# Initialize video capture directly
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture device")
    exit()

# Initialize smoothing variables
smooth_x, smooth_y = None, None

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw the landmarks for visualization
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check if it's the right hand
                if hand_info.classification[0].label == "Right":
                    # Get landmarks
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                    # Check for fist gesture
                    threshold = 0.05
                    is_fist = (
                        calculate_distance(index_tip, index_mcp) < threshold and
                        calculate_distance(middle_tip, middle_mcp) < threshold
                    )

                    if is_fist:
                        # Calculate center of fist
                        center_x = (wrist.x + index_mcp.x + middle_mcp.x) / 3
                        center_y = (wrist.y + index_mcp.y + middle_mcp.y) / 3

                        # Map to screen coordinates
                        screen_x = int(center_x * screen_width)
                        screen_y = int(center_y * screen_height)

                        # Apply smoothing
                        smooth_x = smooth_coords(screen_x, smooth_x)
                        smooth_y = smooth_coords(screen_y, smooth_y)

                        # Move mouse if smoothing is initialized
                        if smooth_x is not None and smooth_y is not None:
                            pyautogui.moveTo(smooth_x, smooth_y)

        # Display the frame
        cv2.imshow('Hand Tracking', frame)
        
        # Break the loop on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()