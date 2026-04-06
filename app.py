import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Sign Language Detection", layout="centered")

st.title("🖐️ Real-Time Sign Language Detection")
st.write("Using Computer Vision and Hand Tracking")

# -------------------------------
# Initialize MediaPipe SAFELY
# -------------------------------
@st.cache_resource
def load_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

hands = load_hands()
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# -------------------------------
# Finger Detection
# -------------------------------
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

# -------------------------------
# Gesture Recognition
# -------------------------------
def recognize_sign(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "A"
    elif fingers == [0, 1, 1, 1, 1]:
        return "B"
    elif fingers == [1, 1, 0, 0, 0]:
        return "L"
    elif fingers == [1, 1, 1, 0, 0]:
        return "W"
    else:
        return "Unknown"

# -------------------------------
# Video Transformer
# -------------------------------
class SignDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process hand detection
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                fingers = get_finger_states(handLms)
                sign = recognize_sign(fingers)

                cv2.putText(img, f"Sign: {sign}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

        return img

# -------------------------------
# Start Webcam (WebRTC)
# -------------------------------
st.subheader("📷 Live Camera Feed")

webrtc_streamer(
    key="sign-detection",
    video_transformer_factory=SignDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)
