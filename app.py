import streamlit as st
import cv2
import mediapipe as mp
from groq import Groq
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -------------------------------
# 🖐️ MEDIAPIPE SETUP
# -------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# ✋ GESTURE FUNCTIONS
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

def recognize(f):
    if f == [0,0,0,0,0]: return "A"
    elif f == [0,1,1,1,1]: return "B"
    elif f == [1,1,0,0,0]: return "L"
    elif f == [1,1,1,0,0]: return "W"
    return ""

# -------------------------------
# 🎥 VIDEO PROCESSOR
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
                letter = recognize(fingers)

                if letter:
                    self.text += letter

                cv2.putText(img, f"{letter}", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.putText(img, f"Text: {self.text}", (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        return img

# -------------------------------
# 🌐 STREAMLIT UI
# -------------------------------
st.title("🖐️ Sign Language Detection + AI")

# 🔑 Manual API Key Input
api_key = st.text_input("Enter your GROQ API Key:", type="password")

# Start webcam
webrtc_ctx = webrtc_streamer(
    key="sign-lang",
    video_processor_factory=VideoProcessor
)

# -------------------------------
# 🤖 AI FUNCTION
# -------------------------------
def enhance_text(text, api_key):
    if not api_key or text.strip() == "":
        return "Enter API key and show gestures"

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Convert these letters into meaningful English words or sentence."},
                {"role": "user", "content": text}
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# 🎯 BUTTON ACTION
# -------------------------------
if webrtc_ctx.video_processor:
    if st.button("✨ Convert to Sentence"):
        text = webrtc_ctx.video_processor.text
        result = enhance_text(text, api_key)

        st.write("### 📝 Detected Letters:")
        st.write(text)

        st.write("### 🤖 AI Output:")
        st.success(result)
