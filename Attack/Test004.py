import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# Constants and thresholds
HISTORY_SIZE = 10  # Number of frames to track
ANGULAR_VELOCITY_THRESHOLD = 30  # Adjust based on experimentation
WRIST_VELOCITY_THRESHOLD = 0.5  # Adjust based on experimentation
TORSO_ORIENTATION_THRESHOLD = 10  # Degrees, to account for acceptable torso motion

# History for storing landmarks
landmark_history = deque(maxlen=HISTORY_SIZE)

def calculate_angle(v1, v2):
    """Calculate the angle (in degrees) between two vectors."""
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude == 0:
        return 0
    return np.degrees(np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0)))

def calculate_velocity(point1, point2, time_interval=1/30):  # Assuming 30 FPS
    """Calculate the velocity between two points."""
    distance = np.linalg.norm(np.array(point2) - np.array(point1))
    return distance / time_interval

def detect_sword_swing(landmark_history):
    """Detect sword swinging action based on landmarks."""
    if len(landmark_history) < 2:
        return False  # Not enough data

    # Get landmarks from the most recent frame
    curr_landmarks = landmark_history[-1]
    prev_landmarks = landmark_history[-2]

    # Calculate angular velocity
    shoulder = np.array(curr_landmarks[12])
    elbow = np.array(curr_landmarks[14])
    wrist = np.array(curr_landmarks[16])
    
    prev_shoulder = np.array(prev_landmarks[12])
    prev_elbow = np.array(prev_landmarks[14])
    prev_wrist = np.array(prev_landmarks[16])
    
    upper_arm = elbow - shoulder
    forearm = wrist - elbow
    prev_upper_arm = prev_elbow - prev_shoulder
    prev_forearm = prev_wrist - prev_elbow

    curr_angle = calculate_angle(upper_arm, forearm)
    prev_angle = calculate_angle(prev_upper_arm, prev_forearm)
    angular_velocity = abs(curr_angle - prev_angle)

    # Calculate wrist velocity
    wrist_velocity = calculate_velocity(prev_wrist, wrist)

    # Calculate torso orientation change
    hip = np.array(curr_landmarks[24])
    torso_vector = shoulder - hip
    prev_torso_vector = prev_shoulder - np.array(prev_landmarks[24])
    curr_torso_orientation = calculate_angle(torso_vector, [1, 0, 0])  # Compare to x-axis
    prev_torso_orientation = calculate_angle(prev_torso_vector, [1, 0, 0])
    torso_orientation_change = abs(curr_torso_orientation - prev_torso_orientation)

    # Detect swing based on thresholds
    if (angular_velocity > ANGULAR_VELOCITY_THRESHOLD and
        wrist_velocity > WRIST_VELOCITY_THRESHOLD and
        torso_orientation_change < TORSO_ORIENTATION_THRESHOLD):
        return True

    return False

# Video capture and detection loop
cap = cv2.VideoCapture(0)  # Use webcam
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
        if detect_sword_swing(landmark_history):
            cv2.putText(frame, "Swing Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Sword Swing Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
