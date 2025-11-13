#video_capture.py
import cv2
import os
from datetime import datetime
import RPi.GPIO as GPIO
import time

# --- SETTINGS ---
VIDEO_DIR = "/media/schlorchi00/78F0F9A6F0F96AB0/DIC_videos"
FPS = 30
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
VIDEO_FORMAT = "h264"

# Make sure the folder exists
os.makedirs(VIDEO_DIR, exist_ok=True)

# --- SETUP GPIO ---
BUTTON_PIN = 11  # Physical pin 11 → GPIO17
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# --- INITIALIZE CAMERA ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

recording = False
out = None
last_button_state = GPIO.input(BUTTON_PIN)
debounce_time = 0.3
last_press_time = time.time()

print("Camera started. Press the button to start/stop recording. ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Check button state with simple debounce
    button_state = GPIO.input(BUTTON_PIN)
    if button_state == 0 and (time.time() - last_press_time) > debounce_time:
        last_press_time = time.time()
        if not recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"video_{timestamp}.{VIDEO_FORMAT}"
            save_path = os.path.join(VIDEO_DIR, filename)
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(save_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            recording = True
            print(f"Started recording: {save_path}")
        else:
            # Stop recording
            recording = False
            out.release()
            out = None
            print("Stopped recording")

    # Write frame if recording
    if recording and out is not None:
        out.write(frame)

    # Display preview with REC indicator
    display_frame = frame.copy()
    if recording:
        cv2.putText(display_frame, "REC", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Camera Preview", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

# Cleanup
if out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
print("Program exited")
