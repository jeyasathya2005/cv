import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Sign Language Detection", layout="centered")

st.title("🖐️ Real-Time Sign Language Detection")
st.write("Using Computer Vision and Hand Tracking")

# -------------------------------
# Initialize MediaPipe
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# Finger Detection Function
# -------------------------------
def get_finger_states(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    states = []

    for i, tip in enumerate(tips):
        if i == 0:
            # Thumb (horizontal)
            states.append(hand_landmarks.landmark[tip].x <
                          hand_landmarks.landmark[tip - 1].x)
        else:
            # Other fingers (vertical)
            states.append(hand_landmarks.landmark[tip].y <
                          hand_landmarks.landmark[tip - 2].y)

    return states

# -------------------------------
# Gesture Recognition Logic
# -------------------------------
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

# -------------------------------
# Video Transformer Class
# -------------------------------
class SignLanguageDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                # Detect fingers
                fingers = get_finger_states(handLms)

                # Recognize sign
                sign = recognize_sign(fingers)

                # Display text
                cv2.putText(img, f"Sign: {sign}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

        return img

# -------------------------------
# Start Webcam
# -------------------------------
st.subheader("📷 Live Camera")

webrtc_streamer(
    key="sign-detection",
    video_transformer_factory=SignLanguageDetector
)
