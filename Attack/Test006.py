import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Constants and thresholds
FPS = 30
HISTORY_SIZE = int(FPS * 1.5)  # 1.5 seconds of history
VERTICAL_MOVEMENT_THRESHOLD = 0.03  # Made more sensitive
SWING_STATE_THRESHOLD = 3  # Frames needed to confirm state
COOLDOWN_FRAMES = 15  # Prevent multiple detections for the same swing

# History and state tracking
landmark_history = deque(maxlen=HISTORY_SIZE)
swing_state = "ready"  # States: ready, lifting, swinging
state_counter = 0
total_swings = 0
cooldown_counter = 0

def get_wrist_height(landmarks):
    """Get normalized wrist height relative to shoulder"""
    if not landmarks:
        return 0
    
    shoulder = np.array([landmarks[12][1]])  # Right shoulder Y coordinate
    wrist = np.array([landmarks[16][1]])     # Right wrist Y coordinate
    return float(shoulder - wrist)  # Positive when wrist is above shoulder

def detect_swing_motion(landmark_history):
    """Detect swing motion using state machine approach"""
    global swing_state, state_counter, total_swings, cooldown_counter
    
    if len(landmark_history) < 2 or cooldown_counter > 0:
        cooldown_counter = max(0, cooldown_counter - 1)
        return False

    current_wrist_height = get_wrist_height(landmark_history[-1])
    prev_wrist_height = get_wrist_height(landmark_history[-2])
    height_change = current_wrist_height - prev_wrist_height

    # State machine logic with improved reliability
    if swing_state == "ready":
        if height_change > VERTICAL_MOVEMENT_THRESHOLD:  # Moving up
            state_counter += 1
            if state_counter >= SWING_STATE_THRESHOLD:
                swing_state = "lifting"
                state_counter = 0
        else:
            state_counter = max(0, state_counter - 1)  # Gradual decrease

    elif swing_state == "lifting":
        if current_wrist_height > 0.1:  # Ensure hand is raised enough
            if height_change < -VERTICAL_MOVEMENT_THRESHOLD:  # Starting to move down
                swing_state = "swinging"
                state_counter = 0
            elif height_change > VERTICAL_MOVEMENT_THRESHOLD:  # Still moving up
                state_counter = 0  # Reset counter but stay in lifting state
        else:
            swing_state = "ready"
            state_counter = 0

    elif swing_state == "swinging":
        if height_change < -VERTICAL_MOVEMENT_THRESHOLD:  # Continuing downward motion
            state_counter += 1
            if state_counter >= SWING_STATE_THRESHOLD:
                swing_state = "ready"
                state_counter = 0
                total_swings += 1
                cooldown_counter = COOLDOWN_FRAMES  # Prevent immediate re-detection
                return True
        else:
            state_counter = max(0, state_counter - 1)  # Gradual decrease

    return False

# Video capture and detection loop
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for a mirrored view
    frame = cv2.flip(frame, 1)

    # Process frame with MediaPipe Pose
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Extract landmarks
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        # Add to history
        landmark_history.append(landmarks)

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Detect swinging action
        swing_detected = detect_swing_motion(landmark_history)

        # Display current state and swing count
        cv2.putText(frame, f"State: {swing_state}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Total Swings: {total_swings}", (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if swing_detected:
            cv2.putText(frame, "SWING!", (50, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow("Swing Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()