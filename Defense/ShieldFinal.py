import numpy as np
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_drawing = mp.solutions.drawing_utils

def is_left_hand(handedness):
    """Check if the detected hand is the left hand"""
    return handedness[0].classification[0].label == "Left"

def is_stop_gesture(landmarks):
    """
    Detect if the hand is in a stop gesture position:
    - Palm facing the camera
    - Fingers extended upward
    - Hand raised above wrist
    """
    # Get key landmarks
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
    index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    middle_tip = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    ring_tip = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
    pinky_tip = np.array([landmarks[20].x, landmarks[20].y, landmarks[20].z])
    
    # Get middle points of fingers
    index_pip = np.array([landmarks[6].x, landmarks[6].y, landmarks[6].z])
    middle_pip = np.array([landmarks[10].x, landmarks[10].y, landmarks[10].z])
    ring_pip = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
    pinky_pip = np.array([landmarks[18].x, landmarks[18].y, landmarks[18].z])
    
    # Check if fingers are extended upward
    finger_heights = [
        index_tip[1] < index_pip[1],  # Check if tip is above middle point
        middle_tip[1] < middle_pip[1],
        ring_tip[1] < ring_pip[1],
        pinky_tip[1] < pinky_pip[1]
    ]
    
    fingers_extended = all(finger_heights)
    
    # Check if hand is raised (all fingertips should be above wrist)
    fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
    hand_raised = all(tip[1] < wrist[1] for tip in fingertips)
    
    # Check if palm is facing the camera using z-coordinates
    palm_center = np.array([landmarks[9].z])  # Middle finger base
    palm_facing = palm_center < landmarks[5].z  # Compare with index finger base
    
    # Combine all conditions
    return fingers_extended and hand_raised and palm_facing

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a temporal smoothing buffer for shield state
shield_buffer = [False] * 5
shield_state = "Down"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    
    # Process the frame
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True
    
    # Initialize current frame's shield detection
    current_shield_up = False
    
    # Process hand landmarks if detected
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Only process left hand
            if is_left_hand(results.multi_handedness):
                # Check for stop gesture
                current_shield_up = is_stop_gesture(hand_landmarks.landmark)
                
                # Draw hand landmarks
                color = (0, 255, 0) if current_shield_up else (255, 0, 0)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2)
                )
    
    # Update shield buffer for temporal smoothing
    shield_buffer.pop(0)
    shield_buffer.append(current_shield_up)
    
    # Update shield state based on majority voting
    if sum(shield_buffer) > len(shield_buffer) / 2:
        shield_state = "Up"
    else:
        shield_state = "Down"
    
    # Display shield state
    color = (0, 255, 0) if shield_state == "Up" else (0, 0, 255)
    cv2.putText(
        frame,
        f"Shield: {shield_state}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )
    
    # Add visual guide for hand position
    cv2.putText(
        frame,
        "Show left hand in 'STOP' position for shield",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    # Display the frame
    cv2.imshow('Shield Detection', frame)
    
    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()