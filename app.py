import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("🖐️ Real-Time Sign Language Detection")
st.write("Using Computer Vision and Hand Tracking")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Finger detection
def get_finger_states(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    states = []

    for i, tip in enumerate(tips):
        if i == 0:
            states.append(hand_landmarks.landmark[tip].x <
                          hand_landmarks.landmark[tip - 1].x)
        else:
            states.append(hand_landmarks.landmark[tip].y <
                          hand_landmarks.landmark[tip - 2].y)

    return states

# Gesture logic
def recognize_sign(fingers):
    if fingers == [0,0,0,0,0]:
        return "A"
    elif fingers == [0,1,1,1,1]:
        return "B"
    elif fingers == [1,1,0,0,0]:
        return "L"
    elif fingers == [1,1,1,0,0]:
        return "W"
    else:
        return "Unknown"

# Webcam start button
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to access camera")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            fingers = get_finger_states(handLms)
            sign = recognize_sign(fingers)

            cv2.putText(frame, f"Sign: {sign}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
