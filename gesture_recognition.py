import cv2
import mediapipe as mp
import pyautogui
import keyboard
import time
import numpy as np


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)


# Function to get finger states

def get_finger_states(hand_landmarks, hand_label):
    finger_states = []

    # Tip and pip landmarks for fingers (index to pinky)
    tips_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    # Check for each finger (excluding thumb)
    for tip, pip in zip(tips_ids, pip_ids):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            finger_states.append(1)  # Finger is open
        else:
            finger_states.append(0)  # Finger is closed

    # Thumb: use x-coordinates, but flip logic for left vs right
    if hand_label == "Right":
        thumb_open = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
    else:  # Left hand
        thumb_open = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x

    return finger_states, int(thumb_open)

last_trigger_time = 0
cooldown = 1  # in seconds

screen_width, screen_height = pyautogui.size()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe

    results = hands.process(rgb_frame)  # Detect hands

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get label ("Left" or "Right")
            hand_label = results.multi_handedness[idx].classification[0].label

            fingers, thumb = get_finger_states(hand_landmarks, hand_label)
            
            # Determine gesture based on finger states
            gesture = "Unknown"

            if all(f == 1 for f in fingers) and thumb == 1:
                gesture = "Open Palm"
            elif all(f == 0 for f in fingers) and thumb == 0:
                gesture = "Fist"
            elif all(f == 0 for f in fingers) and thumb == 1:
                gesture = "Thumbs Up"
            elif fingers == [1, 1, 0, 0] and thumb == 0:
                gesture = "Peace Sign"
            elif fingers == [1, 0, 0, 0] and thumb == 0:
                gesture = "Pointing"
            elif fingers == [0, 0, 0, 1] and thumb == 1:
                gesture = "Call Me"
            elif fingers == [1, 1, 1, 1] and thumb == 0:
                gesture = "Stop Sign"


            if gesture == "Pointing":
                index_finger_tip = hand_landmarks.landmark[8]
                screen_x = int(index_finger_tip.x * screen_width)
                screen_y = int(index_finger_tip.y * screen_height)
                pyautogui.moveTo(screen_x, screen_y)

                thumb_tip = hand_landmarks.landmark[4]
                dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y]))

                if dist < 0.03:
                    pyautogui.click()
                    time.sleep(0.3)
            # ✨ End of mouse control
        
        #Using gestures for various actions
        current_time = time.time()
        if current_time - last_trigger_time > cooldown:
            if gesture == "Thumbs Up":
                pyautogui.press("volumeup")
                last_trigger_time = current_time

            elif gesture == "Fist":
                pyautogui.press("volumedown")
                last_trigger_time = current_time

            elif gesture == "Call Me":
                pyautogui.press("volumemute")
                last_trigger_time = current_time

            elif gesture == "Open Palm":
                keyboard.press_and_release("play/pause media")
                last_trigger_time = current_time

            elif gesture == "Peace Sign":
                keyboard.press_and_release("next track")
                last_trigger_time = current_time

            elif gesture == "Pointing":
                pyautogui.screenshot("gesture_screenshot.png")
                print("Screenshot saved!")
                last_trigger_time = current_time
    

            # Display the gesture and hand label
            cv2.putText(frame, f"{gesture} ({hand_label})", (10, 50 + idx * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
