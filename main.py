import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import pyautogui
import screen_brightness_control as sbc

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not working")
    exit()

prev_vol = 0
prev_bright = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks and results.multi_handedness:
        for handLms, handType in zip(results.multi_hand_landmarks, results.multi_handedness):

            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]

                length = hypot(x2 - x1, y2 - y1)
                value = np.interp(length, [30, 200], [0, 100])

                hand_label = handType.classification[0].label

                # 👉 RIGHT HAND → VOLUME
                if hand_label == "Right":
                    if value > prev_vol + 5:
                        pyautogui.press("volumeup")
                    elif value < prev_vol - 5:
                        pyautogui.press("volumedown")

                    prev_vol = value

                    cv2.putText(img, f'Volume: {int(value)}%', (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # 👉 LEFT HAND → BRIGHTNESS
                elif hand_label == "Left":
                    if value > prev_bright + 5:
                        sbc.set_brightness(min(100, int(value)))
                    elif value < prev_bright - 5:
                        sbc.set_brightness(max(0, int(value)))

                    prev_bright = value

                    cv2.putText(img, f'Brightness: {int(value)}%', (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    cv2.imshow("AI Gesture Control System", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
