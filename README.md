**Gesture-Based Video Effects using OpenCV & MediaPipe**

This project uses OpenCV and MediaPipe to detect hand gestures via a webcam and apply real-time video effects based on the number of fingers raised.🖐️

📌 Features 
- Hand detection and tracking using MediaPipe. 
- Finger Count Estimation.

Real-time Effects Based on Finger Count 🎥: 
- 0 fingers - Zoom In 🔍 (Crops the frame to create a zoom effect)
- 1 finger - Zoom Out 🔍 (Shrinks the frame and adds borders)
- 2 fingers - Grayscale Mode ⚫⚪ (Converts video to black and white)
- 3 fingers - Blur Effect 🌫️ (Applies a soft blur to the frame)
- 4 fingers - Inverted Colors 🎨 (Inverts the colors for a unique effect)
- 5 fingers - Reset View 🔄 (Restores the normal video feed)

🛠️ Requirements 
Make sure you have the following installed: 
- Python 3.x 
- OpenCV 
- MediaPipe 
- NumPy

Install dependencies with: 
pip install opencv-python mediapipe numpy

🖼️ How It Works 
- The script captures video from your webcam.🎥 
- It detects hand landmarks using MediaPipe. 
- The number of fingers raised is counted.✋ 
- Based on the count, an effect is applied.
