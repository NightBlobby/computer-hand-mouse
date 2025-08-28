import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_w, screen_h = pyautogui.size()

# Gesture state
last_pinch_time = 0
pinch_active = False
last_hand_pos = None
last_swipe_time = 0

# Cursor smoothing
cursor_x, cursor_y = 0, 0
smoothing = 5  # higher = smoother, but slower response


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def is_pinch(landmarks):
    # Check distance between thumb tip (4) and index tip (8)
    thumb = (landmarks[4].x, landmarks[4].y)
    index = (landmarks[8].x, landmarks[8].y)
    return distance(thumb, index) < 0.05  # tuned threshold


def detect_swipe(curr_pos, prev_pos):
    global last_swipe_time
    if prev_pos is None:
        return None
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    dist = np.hypot(dx, dy)

    if dist > 0.25 and (time.time() - last_swipe_time) > 0.7:  # swipe speed
        last_swipe_time = time.time()
        if abs(dx) > abs(dy):
            return "LEFT" if dx < 0 else "RIGHT"
        else:
            return "UP" if dy < 0 else "DOWN"
    return None


# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            landmarks = hand_landmarks.landmark

            # Get index finger tip position
            index_x = int(landmarks[8].x * w)
            index_y = int(landmarks[8].y * h)

            # Smooth cursor movement
            target_x = landmarks[8].x * screen_w
            target_y = landmarks[8].y * screen_h
            cursor_x += (target_x - cursor_x) / smoothing
            cursor_y += (target_y - cursor_y) / smoothing
            pyautogui.moveTo(cursor_x, cursor_y)

            # Pinch click detection
            if is_pinch(landmarks):
                if not pinch_active:
                    current_time = time.time()
                    if current_time - last_pinch_time < 0.4:
                        pyautogui.click(button="right")  # double pinch = right click
                        print("Right Click!")
                        last_pinch_time = 0
                    else:
                        pyautogui.click(button="left")  # single pinch
                        print("Left Click!")
                        last_pinch_time = current_time
                pinch_active = True
            else:
                pinch_active = False

            # Swipe detection
            curr_pos = (landmarks[0].x, landmarks[0].y)
            gesture = detect_swipe(curr_pos, last_hand_pos)
            last_hand_pos = curr_pos

            if gesture == "DOWN":
                pyautogui.hotkey("win", "d")  # hide all tabs
                print("Swipe Down -> Hide Windows")
            elif gesture == "UP":
                pyautogui.scroll(800)
                print("Swipe Up -> Scroll Up")
            elif gesture == "LEFT":
                pyautogui.hotkey("alt", "tab")
                print("Swipe Left -> Switch App")
            elif gesture == "RIGHT":
                pyautogui.hotkey("ctrl", "tab")
                print("Swipe Right -> Next Tab")

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
