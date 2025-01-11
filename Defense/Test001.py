import numpy as np
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1  # Only track one hand for better performance
)
mp_drawing = mp.solutions.drawing_utils

def is_left_hand(hand_landmarks, handedness):
    """Check if the detected hand is the left hand"""
    return handedness[0].classification[0].label == "Left"

def is_palm_open(hand_landmarks):
    """
    Detect if the palm is open by measuring finger extension and orientation
    """
    # Get key landmarks
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
    index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
    middle_tip = np.array([hand_landmarks[12].x, hand_landmarks[12].y])
    ring_tip = np.array([hand_landmarks[16].x, hand_landmarks[16].y])
    pinky_tip = np.array([hand_landmarks[20].x, hand_landmarks[20].y])
    
    # Get finger base points
    index_base = np.array([hand_landmarks[5].x, hand_landmarks[5].y])
    middle_base = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
    ring_base = np.array([hand_landmarks[13].x, hand_landmarks[13].y])
    pinky_base = np.array([hand_landmarks[17].x, hand_landmarks[17].y])
    
    # Calculate finger extensions (distance from base to tip)
    index_extension = np.linalg.norm(index_tip - index_base)
    middle_extension = np.linalg.norm(middle_tip - middle_base)
    ring_extension = np.linalg.norm(ring_tip - ring_base)
    pinky_extension = np.linalg.norm(pinky_tip - pinky_base)
    
    # Calculate average finger length for normalization
    avg_finger_length = np.mean([index_extension, middle_extension, ring_extension, pinky_extension])
    
    # Extension thresholds (relative to average finger length)
    extension_threshold = 0.7
    
    # Check if fingers are extended
    fingers_extended = (
        index_extension > avg_finger_length * extension_threshold and
        middle_extension > avg_finger_length * extension_threshold and
        ring_extension > avg_finger_length * extension_threshold and
        pinky_extension > avg_finger_length * extension_threshold
    )
    
    return fingers_extended

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a temporal smoothing buffer for shield state
shield_buffer = [False] * 5  # Store last 5 frames
shield_state = "Down"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True
    
    # Initialize current frame's shield detection
    current_shield_up = False
    
    # Process hand landmarks if detected
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Only process if it's the left hand
            if is_left_hand(hand_landmarks.landmark, results.multi_handedness):
                # Check if palm is open
                current_shield_up = is_palm_open(hand_landmarks.landmark)
                
                # Draw hand landmarks with different colors based on shield state
                color = (0, 255, 0) if current_shield_up else (255, 0, 0)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2)
                )
    
    # Update shield buffer
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
    
    # Display the frame
    cv2.imshow('Shield Detection', frame)
    
    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()