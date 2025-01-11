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
HISTORY_SIZE = int(FPS * 1.5)  # 1.5 seconds of history to capture full swing motion
SWING_DURATION = int(FPS * 1)   # 1 second for complete swing
VERTICAL_MOVEMENT_THRESHOLD = 0.1  # Threshold for vertical movement
SWING_STATE_THRESHOLD = 5  # Minimum frames to confirm each swing state

# History for storing landmarks and swing detection
landmark_history = deque(maxlen=HISTORY_SIZE)
swing_state = "ready"  # States: ready, lifting, swinging
state_counter = 0

def get_wrist_height(landmarks):
    """Get normalized wrist height relative to shoulder"""
    if not landmarks:
        return 0
    
    shoulder = np.array([landmarks[12][1]])  # Right shoulder Y coordinate
    wrist = np.array([landmarks[16][1]])     # Right wrist Y coordinate
    return float(shoulder - wrist)  # Positive when wrist is above shoulder

def detect_swing_motion(landmark_history):
    """Detect swing motion using state machine approach"""
    global swing_state, state_counter
    
    if len(landmark_history) < 2:
        return False

    current_wrist_height = get_wrist_height(landmark_history[-1])
    prev_wrist_height = get_wrist_height(landmark_history[-2])
    height_change = current_wrist_height - prev_wrist_height

    # State machine logic
    if swing_state == "ready":
        if height_change > VERTICAL_MOVEMENT_THRESHOLD:  # Moving up
            state_counter += 1
            if state_counter >= SWING_STATE_THRESHOLD:
                swing_state = "lifting"
                state_counter = 0
        else:
            state_counter = 0

    elif swing_state == "lifting":
        if current_wrist_height > VERTICAL_MOVEMENT_THRESHOLD:  # Wrist is high
            if height_change < -VERTICAL_MOVEMENT_THRESHOLD:  # Starting to move down
                swing_state = "swinging"
                state_counter = 0
        else:
            swing_state = "ready"
            state_counter = 0

    elif swing_state == "swinging":
        if height_change < -VERTICAL_MOVEMENT_THRESHOLD:  # Continuing downward motion
            state_counter += 1
            if state_counter >= SWING_STATE_THRESHOLD:
                swing_state = "ready"
                state_counter = 0
                return True  # Swing detected
        else:
            swing_state = "ready"
            state_counter = 0

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

        # Display current state and detection
        cv2.putText(frame, f"State: {swing_state}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if swing_detected:
            cv2.putText(frame, "SWING DETECTED!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Swing Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()