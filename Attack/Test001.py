import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Line coordinates
line1_y = 200  # Y-coordinate of the upper horizontal line
line2_y = 600  # Y-coordinate of the lower horizontal line
movement_threshold = 0.2  # Time threshold for a quick movement (seconds)

# Initialize OpenCV
cap = cv2.VideoCapture(0)  # Use webcam

# Increase resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_y = None
attack_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(rgb_frame)

    # Draw horizontal lines
    cv2.line(frame, (0, line1_y), (frame.shape[1], line1_y), (0, 255, 0), 2)
    cv2.line(frame, (0, line2_y), (frame.shape[1], line2_y), (0, 255, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Detect only the right hand
            handedness = result.multi_handedness[0].classification[0].label
            if handedness != "Right":
                continue

            # Get the coordinates of the fist (use wrist as a proxy for simplicity)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_y = int(wrist.y * frame.shape[0])

            # Check for movement across lines
            if prev_y is not None:
                if (prev_y < line1_y and wrist_y > line2_y) or (prev_y > line2_y and wrist_y < line1_y):
                    current_time = time.time()
                    if attack_time is None or (current_time - attack_time > movement_threshold):
                        print("Attack detected!")
                        cv2.putText(frame, "Attack!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        attack_time = current_time

            prev_y = wrist_y

            # Draw landmarks on the hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Attack Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
