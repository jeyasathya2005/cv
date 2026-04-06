import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from groq import Groq
import os

# -------------------------------
# 🎨 UI CONFIG
# -------------------------------
st.set_page_config(page_title="SignSpeak CV", layout="wide")

st.title("🖐️ Real-Time Sign Language Detection")
st.caption("Vision → Brain → AI Output")

# -------------------------------
# 🔑 API KEY INPUT
# -------------------------------
api_key = st.sidebar.text_input("Enter GROQ API Key", type="password")

# -------------------------------
# 🖐️ MEDIAPIPE SETUP
# -------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# ✋ HAND → LETTER MODULE
# -------------------------------
def get_fingers(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    for i, tip in enumerate(tips):
        if i == 0:
            fingers.append(hand.landmark[tip].x < hand.landmark[tip - 1].x)
        else:
            fingers.append(hand.landmark[tip].y < hand.landmark[tip - 2].y)

    return fingers

def recognize_sign(f):
    if f == [0,0,0,0,0]: return "A"
    elif f == [0,1,1,1,1]: return "B"
    elif f == [1,1,0,0,0]: return "L"
    elif f == [1,1,1,0,0]: return "W"
    return ""

# -------------------------------
# 🎥 VISION MODULE (HAND TRACKING)
# -------------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=1)
        self.text = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

                fingers = get_fingers(hand)
                letter = recognize_sign(fingers)

                if letter:
                    self.text += letter

                cv2.putText(img, f"{letter}", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.putText(img, f"Text: {self.text}", (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        return img

# -------------------------------
# 🧠 BRAIN MODULE (GROQ AI)
# -------------------------------
def process_text_with_ai(text, api_key):
    if not api_key:
        return "Enter API key"
    if text.strip() == "":
        return "No gesture detected"

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Convert these letters into meaningful English sentence."},
                {"role": "user", "content": text}
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {e}"

# -------------------------------
# 🌐 MAIN UI FLOW
# -------------------------------
st.subheader("1️⃣ Vision: Hand Detection")

webrtc_ctx = webrtc_streamer(
    key="sign-detect",
    video_processor_factory=VideoProcessor
)

# -------------------------------
# 🧠 AI OUTPUT
# -------------------------------
if webrtc_ctx.video_processor:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2️⃣ Detected Letters")
        detected_text = webrtc_ctx.video_processor.text
        st.write(detected_text)

    with col2:
        st.subheader("3️⃣ AI Output")

        if st.button("✨ Convert to Sentence"):
            result = process_text_with_ai(detected_text, api_key)
            st.success(result)
