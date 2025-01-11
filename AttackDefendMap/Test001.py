import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# IP Webcam URL

# Initialize MediaPipe Pose for attack detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands for defend detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Constants for attack detection
FPS = 30
HISTORY_SIZE = int(FPS * 1.5)
VERTICAL_MOVEMENT_THRESHOLD = 0.03
SWING_STATE_THRESHOLD = 2
COOLDOWN_FRAMES = 8
SHOULDER_HEIGHT_THRESHOLD = 0.05

# History and state tracking
landmark_history = deque(maxlen=HISTORY_SIZE)
swing_state = "ready"
state_counter = 0
total_swings = 0
cooldown_counter = 0

# Shield detection buffer
shield_buffer = [False] * 5
shield_state = "Down"

def get_wrist_position_relative_to_shoulder(landmarks):
    if not landmarks:
        return 0
    right_shoulder = np.array([landmarks[11][1]])
    right_wrist = np.array([landmarks[16][1]])
    return float(right_shoulder - right_wrist)

def detect_swing_motion(landmark_history):
    global swing_state, state_counter, total_swings, cooldown_counter
    if len(landmark_history) < 2 or cooldown_counter > 0:
        cooldown_counter = max(0, cooldown_counter - 1)
        return False
    current_position = get_wrist_position_relative_to_shoulder(landmark_history[-1])
    prev_position = get_wrist_position_relative_to_shoulder(landmark_history[-2])
    position_change = current_position - prev_position

    if swing_state == "ready":
        if position_change > VERTICAL_MOVEMENT_THRESHOLD:
            state_counter += 1
            if state_counter >= SWING_STATE_THRESHOLD:
                swing_state = "lifting"
                state_counter = 0
        else:
            state_counter = 0

    elif swing_state == "lifting":
        if -SHOULDER_HEIGHT_THRESHOLD <= current_position <= 0.2:
            if position_change < -VERTICAL_MOVEMENT_THRESHOLD:
                swing_state = "swinging"
                state_counter = 0
        else:
            swing_state = "ready"
            state_counter = 0

    elif swing_state == "swinging":
        if position_change < -VERTICAL_MOVEMENT_THRESHOLD:
            state_counter += 1
            if state_counter >= SWING_STATE_THRESHOLD:
                swing_state = "ready"
                state_counter = 0
                total_swings += 1
                cooldown_counter = COOLDOWN_FRAMES
                return True
        else:
            state_counter = 0
    return False

def is_left_hand(handedness):
    return handedness[0].classification[0].label == "Left"

def is_stop_gesture(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
    index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    middle_tip = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    ring_tip = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
    pinky_tip = np.array([landmarks[20].x, landmarks[20].y, landmarks[20].z])
    index_pip = np.array([landmarks[6].x, landmarks[6].y, landmarks[6].z])
    middle_pip = np.array([landmarks[10].x, landmarks[10].y, landmarks[10].z])
    ring_pip = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
    pinky_pip = np.array([landmarks[18].x, landmarks[18].y, landmarks[18].z])

    finger_heights = [
        index_tip[1] < index_pip[1],
        middle_tip[1] < middle_pip[1],
        ring_tip[1] < ring_pip[1],
        pinky_tip[1] < pinky_pip[1]
    ]
    fingers_extended = all(finger_heights)
    fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
    hand_raised = all(tip[1] < wrist[1] for tip in fingertips)
    palm_center = np.array([landmarks[9].z])
    palm_facing = palm_center < landmarks[5].z

    return fingers_extended and hand_raised and palm_facing

#cap = cv2.VideoCapture(IP_WEBCAM_URL)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb_frame)
    results_hands = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    current_shield_up = False
    if results_pose.pose_landmarks:
        landmarks = [[lm.x, lm.y, lm.z] for lm in results_pose.pose_landmarks.landmark]
        landmark_history.append(landmarks)
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        swing_detected = detect_swing_motion(landmark_history)
        cv2.putText(frame, f"State: {swing_state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Total Swings: {total_swings}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if swing_detected:
            cv2.putText(frame, "SWING!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            if is_left_hand(results_hands.multi_handedness):
                current_shield_up = is_stop_gesture(hand_landmarks.landmark)
                color = (0, 255, 0) if current_shield_up else (255, 0, 0)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=color, thickness=2),
                                          mp_drawing.DrawingSpec(color=color, thickness=2))

    shield_buffer.pop(0)
    shield_buffer.append(current_shield_up)
    shield_state = "Up" if sum(shield_buffer) > len(shield_buffer) / 2 else "Down"
    color = (0, 255, 0) if shield_state == "Up" else (0, 0, 255)
    cv2.putText(frame, f"Shield: {shield_state}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Attack and Defend Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
