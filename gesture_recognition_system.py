# Import required libraries
import cv2  # OpenCV for computer vision tasks
import mediapipe as mp  # MediaPipe for hand tracking and landmark detection
import numpy as np  # NumPy for numerical operations

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands  # Load the Hands module
mp_drawing = mp.solutions.drawing_utils  # Load utility for drawing hand landmarks
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)  # Configure hand tracking model

# Open the default camera (webcam) for capturing video
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    """
    Counts the number of raised fingers based on hand landmark positions.

    Args:
        hand_landmarks (HandLandmark): Contains 21 hand landmark points.

    Returns:
        int: The count of raised fingers.
    """

    # Indices of finger tips and their respective bases in the landmark array
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, and Pinky fingertips
    finger_bases = [6, 10, 14, 18]  # Base points for each finger

    count = 0  # Counter for raised fingers

    # Check if each finger is raised (i.e., tip is above the base in the y-axis)
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            count += 1  # Finger is considered raised

    # Thumb detection (works differently due to its orientation)
    thumb_tip = hand_landmarks.landmark[4]   # Thumb tip
    thumb_ip = hand_landmarks.landmark[3]    # Interphalangeal joint
    thumb_mcp = hand_landmarks.landmark[2]   # Metacarpophalangeal joint

    # Check if thumb is extended by measuring x-axis distance between thumb tip & MCP joint
    if abs(thumb_tip.x - thumb_mcp.x) > 0.07:
        count += 1  # Thumb is considered raised

    return count  # Return the number of fingers raised

def apply_effect(frame, fingers_count):
    """
    Applies visual effects based on the number of raised fingers.

    Args:
        frame (numpy.ndarray): The current frame from the webcam.
        fingers_count (int): Number of fingers raised.

    Returns:
        numpy.ndarray: The modified frame with the applied effect.
    """

    h, w, _ = frame.shape  # Get frame dimensions (height, width, and channels)

    if fingers_count == 0:
        # Zoom in by cropping the center region of the frame
        zoom_factor = 1.5  # Scale factor for zoom
        new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)  # Calculate new dimensions
        x1, y1 = (w - new_w) // 2, (h - new_h) // 2  # Determine cropping region
        frame = frame[y1:y1+new_h, x1:x1+new_w]  # Crop frame
        frame = cv2.resize(frame, (w, h))  # Resize back to original dimensions
    
    elif fingers_count == 1:
        # Zoom out effect by shrinking the frame and adding black borders
        zoom_factor = 0.7  # Scale factor for zooming out
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)  # Shrunk frame dimensions
        frame_small = cv2.resize(frame, (new_w, new_h))  # Shrink the frame
        frame = np.zeros_like(frame)  # Create a black canvas of the original size
        x_offset = (w - new_w) // 2  # X-axis offset for centering
        y_offset = (h - new_h) // 2  # Y-axis offset for centering
        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_small  # Overlay resized frame onto black canvas
    
    elif fingers_count == 2:
        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR format
    
    elif fingers_count == 3:
        # Apply Gaussian blur to the frame
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
    
    elif fingers_count == 4:
        # Apply color inversion effect
        frame = cv2.bitwise_not(frame)
    
    elif fingers_count == 5:
        # Reset view (no effect applied)
        pass

    return frame  # Return the modified frame

# Main loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Exit if there's an issue capturing frames

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB (MediaPipe expects RGB)
    results = hands.process(rgb_frame)  # Process frame for hand detection

    fingers_count = 0  # Initialize fingers count

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Count the number of fingers raised
            fingers_count = count_fingers(hand_landmarks)

    # Apply effect based on the detected number of fingers
    frame = apply_effect(frame, fingers_count)

    # Display the number of fingers detected on the frame
    cv2.putText(frame, f"Fingers: {fingers_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the processed frame in a window
    cv2.imshow("Gesture Recognition", frame)

    # Exit loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()  # Close the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
