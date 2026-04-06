import cv2
import mediapipe as mp
from groq import Groq

# -------------------------------
# GROQ SETUP
# -------------------------------
client = Groq(api_key="YOUR_API_KEY_HERE")

def enhance_text(text):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Convert letters into meaningful word or sentence."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except:
        return text

# -------------------------------
# MEDIAPIPE SETUP
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# GESTURE DETECTION
# -------------------------------
def get_fingers(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    for i, tip in enumerate(tips):
        if i == 0:
            fingers.append(hand.landmark[tip].x <
                           hand.landmark[tip - 1].x)
        else:
            fingers.append(hand.landmark[tip].y <
                           hand.landmark[tip - 2].y)
    return fingers

def recognize(f):
    if f == [0,0,0,0,0]: return "A"
    if f == [0,1,1,1,1]: return "B"
    if f == [1,1,0,0,0]: return "L"
    if f == [1,1,1,0,0]: return "W"
    return ""

# -------------------------------
# MAIN
# -------------------------------
cap = cv2.VideoCapture(0)
sentence = ""

while True:
    _, img = cap.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            fingers = get_fingers(hand)
            letter = recognize(fingers)

            if letter:
                sentence += letter

            cv2.putText(img, f"Letter: {letter}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    # Press SPACE to send to GROQ
    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        output = enhance_text(sentence)
        print("AI Output:", output)
        sentence = ""

    if key == 27:
        break

    cv2.imshow("Sign Detection", img)

cap.release()
cv2.destroyAllWindows()
