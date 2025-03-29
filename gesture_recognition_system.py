import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Open video capture
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    """Accurately count raised fingers using landmark positions."""
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    finger_bases = [6, 10, 14, 18]

    count = 0
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:  
            count += 1

    # Thumb detection
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]

    if abs(thumb_tip.x - thumb_mcp.x) > 0.07:  # Detect thumb movement
        count += 1

    return count

def apply_effect(frame, fingers_count):
    """Apply visual effects based on the number of fingers detected."""
    h, w, _ = frame.shape

    if fingers_count == 0:
        # Zoom in by cropping the center region
        zoom_factor = 1.5
        new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
        x1, y1 = (w - new_w) // 2, (h - new_h) // 2
        frame = frame[y1:y1+new_h, x1:x1+new_w]
        frame = cv2.resize(frame, (w, h))  # Resize back to original
    elif fingers_count == 1:
        # Zoom out by shrinking the frame and adding borders
        zoom_factor = 0.7
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        frame_small = cv2.resize(frame, (new_w, new_h))
        frame = np.zeros_like(frame)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_small
    elif fingers_count == 2:
        # Apply Grayscale filter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back for consistency
    elif fingers_count == 3:
        # Apply Blur effect
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
    elif fingers_count == 4:
        # Apply Color inversion
        frame = cv2.bitwise_not(frame)
    elif fingers_count == 5:
        # Reset view (do nothing)
        pass

    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    fingers_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_count = count_fingers(hand_landmarks)

    # Apply effect based on finger count
    frame = apply_effect(frame, fingers_count)

    # Display fingers count
    cv2.putText(frame, f"Fingers: {fingers_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
