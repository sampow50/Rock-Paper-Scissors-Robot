#run these in the terminal
#py -3.11 -m venv mp_env
#mp_env\Scripts\activate
#python -m pip install --upgrade pip
#pip install mediapipe 
#pip install opencv-python
#python rps_game.py




import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,              # faster
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

gestures = ['Rock', 'Paper', 'Scissors']
rules = {
    ('Rock', 'Scissors'): 'Player wins!',
    ('Scissors', 'Paper'): 'Player wins!',
    ('Paper', 'Rock'): 'Player wins!',
    ('Scissors', 'Rock'): 'Computer wins!',
    ('Paper', 'Scissors'): 'Computer wins!',
    ('Rock', 'Paper'): 'Computer wins!',
    ('Rock', 'Rock'): "It's a draw!",
    ('Paper', 'Paper'): "It's a draw!",
    ('Scissors', 'Scissors'): "It's a draw!"
}

player_score = 0
computer_score = 0

last_round_time = 0.0
round_cooldown = 1.0  # seconds

# MediaPipe landmark indices
TIP = [4, 8, 12, 16, 20]
PIP = [2, 6, 10, 14, 18]  # for thumb: compare tip to knuckle-ish; for others: tip vs pip

def get_hand_gesture(image):
    # Resize can improve consistency and speed
    image = cv2.resize(image, (640, 480))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    if not results.multi_hand_landmarks:
        return None, image

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    lm = hand_landmarks.landmark

    # Simple finger-up test for index/middle/ring/pinky:
    fingers_up = []
    for tip_idx, pip_idx in zip(TIP[1:], PIP[1:]):
        fingers_up.append(lm[tip_idx].y < lm[pip_idx].y)

    # Decide gesture (very basic)
    if all(fingers_up):
        return 'Paper', image
    if fingers_up[0] and fingers_up[1] and (not fingers_up[2]) and (not fingers_up[3]):
        return 'Scissors', image
    if not any(fingers_up):
        return 'Rock', image

    return None, image

# UI state (so text persists between rounds)
player_gesture = "-"
computer_gesture = "-"
result_text = "Show Rock/Paper/Scissors"
# -------------------------

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read camera frame. Try a different camera index (0/1/2).")
        break

    gesture, annotated = get_hand_gesture(frame)

    now = time.time()
    # Only play a round if:
    # - we detected a gesture
    # - cooldown has passed since last round
    if gesture is not None and (now - last_round_time) >= round_cooldown:
        player_gesture = gesture
        computer_gesture = random.choice(gestures)
        result_text = rules.get((player_gesture, computer_gesture), 'Invalid gestures')

        if result_text == 'Player wins!':
            player_score += 1
        elif result_text == 'Computer wins!':
            computer_score += 1

        last_round_time = now

    # Overlay text every frame (no sleeping)
    overlay = [
        (f"Player: {player_gesture}", 30),
        (f"Computer: {computer_gesture}", 70),
        (result_text, 110),
        (f"Player Score: {player_score}", 150),
        (f"Computer Score: {computer_score}", 190),
        ("Press q to quit", 230),
    ]
    for text, y in overlay:
        cv2.putText(annotated, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Rock Paper Scissors', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
