import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("🖐️ Sign Language Detection (Cloud Compatible)")
st.write("Basic gesture detection without MediaPipe")

# -------------------------------
# Simple Gesture Detection (Contour Based)
# -------------------------------
def detect_hand_gesture(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return frame, "No Hand"

    cnt = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(cnt)

    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
    cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)

    # Simple heuristic
    area = cv2.contourArea(cnt)

    if area < 2000:
        gesture = "Far Hand"
    elif area < 5000:
        gesture = "A"
    elif area < 10000:
        gesture = "B"
    else:
        gesture = "Open Hand"

    return frame, gesture


# -------------------------------
# WebRTC Video Class
# -------------------------------
class GestureDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        processed, gesture = detect_hand_gesture(img)

        cv2.putText(processed, f"Gesture: {gesture}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2)

        return processed


# -------------------------------
# Start Camera
# -------------------------------
webrtc_streamer(
    key="gesture-detection",
    video_transformer_factory=GestureDetector,
    media_stream_constraints={"video": True, "audio": False},
)
