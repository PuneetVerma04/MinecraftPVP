import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Screen Dimensions
screen_width, screen_height = pyautogui.size()

# Function to calculate distance
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror-like view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand Detection
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Check if it's the right hand
            if hand_info.classification[0].label == "Right":
                # Get necessary landmarks
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                # Detect if it's a fist
                threshold = 0.05  # Adjust based on testing
                is_fist = (
                    calculate_distance(index_tip, index_mcp) < threshold and
                    calculate_distance(middle_tip, middle_mcp) < threshold
                )

                if is_fist:
                    # Calculate the center of the fist
                    center_x = (wrist.x + index_mcp.x + middle_mcp.x) / 3
                    center_y = (wrist.y + index_mcp.y + middle_mcp.y) / 3

                    # Map to screen coordinates
                    screen_x = int(center_x * screen_width)
                    screen_y = int(center_y * screen_height)

                    # Move Mouse
                    pyautogui.moveTo(screen_x, screen_y)

    # Display Feed (Optional)
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc'
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
