import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

TIP = [4, 8, 12, 16, 20]
PIP = [2, 6, 10, 14, 18]

class HandGestureInput:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def detect_gesture(self, frame, draw=True):
        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None, frame

        hand_landmarks = results.multi_hand_landmarks[0]
        if draw:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        lm = hand_landmarks.landmark

        # index/middle/ring/pinky up?
        fingers_up = []
        for tip_idx, pip_idx in zip(TIP[1:], PIP[1:]):
            fingers_up.append(lm[tip_idx].y < lm[pip_idx].y)

        if all(fingers_up):
            return "Paper", frame
        if fingers_up[0] and fingers_up[1] and (not fingers_up[2]) and (not fingers_up[3]):
            return "Scissors", frame
        if not any(fingers_up):
            return "Rock", frame

        return None, frame

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
