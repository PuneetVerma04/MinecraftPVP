import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
import win32api
import win32con
import keyboard

@dataclass
class SensitivitySettings:
    x: float = 15.0
    y: float = 15.0
    acceleration: float = 1.1
    dead_zone: float = 0.03
    movement_scale: float = 30.0
    max_movement: float = 50.0

class HeadTracker:
    FOREHEAD_TOP = 10  # Example landmark, adjust if needed
    CHIN_BOTTOM = 152  # Example landmark, adjust if needed

    def __init__(self, smoothing_frames=3, calibration_frames=30):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )

        self.sensitivity = SensitivitySettings()
        self.position_history = deque(maxlen=smoothing_frames)
        self.frame_times = deque(maxlen=30)

        self.calibration_center = None
        self.calibration_frames = 0
        self.CALIBRATION_DURATION = calibration_frames
        self.last_position = None
        self.is_calibrating = False

    def recenter_view(self, current_pos):
        self.calibration_center = current_pos
        self.last_position = current_pos
        self.position_history.clear()

    def calculate_head_center(self, landmarks):
        """Calculate the center point between the forehead and chin."""
        forehead = np.array([landmarks[self.FOREHEAD_TOP].x, landmarks[self.FOREHEAD_TOP].y])
        chin = np.array([landmarks[self.CHIN_BOTTOM].x, landmarks[self.CHIN_BOTTOM].y])
        return (forehead + chin) / 2

    def move_camera(self, dx: int, dy: int):
        dx = np.clip(dx, -self.sensitivity.max_movement, self.sensitivity.max_movement)
        dy = np.clip(dy, -self.sensitivity.max_movement, self.sensitivity.max_movement)

        if abs(dx) > 0 or abs(dy) > 0:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

    def calculate_relative_movement(self, current_pos):
        if self.last_position is None:
            self.last_position = current_pos
            return 0, 0

        x_offset = (current_pos[0] - self.calibration_center[0])
        y_offset = (current_pos[1] - self.calibration_center[1])

        if abs(x_offset) < self.sensitivity.dead_zone:
            x_offset = 0
        if abs(y_offset) < self.sensitivity.dead_zone:
            y_offset = 0

        x_movement = np.sign(x_offset) * (abs(x_offset) ** self.sensitivity.acceleration)
        y_movement = np.sign(y_offset) * (abs(y_offset) ** self.sensitivity.acceleration)

        x_movement *= self.sensitivity.x * self.sensitivity.movement_scale
        y_movement *= self.sensitivity.y * self.sensitivity.movement_scale

        self.last_position = current_pos

        return x_movement, y_movement

    def process_frame(self, frame):
        start_time = time.time()

        if keyboard.is_pressed('r'):
            self.is_calibrating = True

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            head_center = self.calculate_head_center(face_landmarks.landmark)

            if self.calibration_frames < self.CALIBRATION_DURATION or self.is_calibrating:
                if self.calibration_center is None or self.is_calibrating:
                    self.calibration_center = head_center
                    self.is_calibrating = False
                else:
                    self.calibration_center = self.calibration_center * 0.8 + head_center * 0.2
                self.calibration_frames += 1
            else:
                dx, dy = self.calculate_relative_movement(head_center)
                if dx != 0 or dy != 0:
                    self.move_camera(dx, dy)

        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)

        return frame

def main():
    print("\nGame Camera Control:")
    print("Press 'R' to recenter, 'Q' to exit\n")

    tracker = HeadTracker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1360)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 786)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = tracker.process_frame(frame)
            cv2.imshow('Game Camera Control', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
