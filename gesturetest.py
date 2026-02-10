import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)  # Speech speed

# Load model, scaler, and class indices
try:
    model = tf.keras.models.load_model("hand_gesture_model_optimized.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("class_indices.pkl", "rb") as f:
        class_indices = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Missing file {e}")
    exit(1)

# Reverse class indices for label lookup
index_to_class = {v: k for k, v in class_indices.items()}

# Constants
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for predictions
TEXT_FILE = "recognized_text.txt"  # File to save recognized text
FRAME_SKIP = 2  # Process every 2nd frame to reduce CPU load
PREFERRED_CAMERA_WIDTH, PREFERRED_CAMERA_HEIGHT = 1280, 720  # Preferred camera feed size
FALLBACK_CAMERA_WIDTH, FALLBACK_CAMERA_HEIGHT = 640, 480  # Fallback size
PANEL_WIDTH, PANEL_HEIGHT = 1280, 400  # Control panel size
BUTTON_WIDTH, BUTTON_HEIGHT = 80, 40
BUTTON_COLOR = (200, 200, 200)  # Light grey
EXIT_BUTTON_COLOR = (0, 0, 255)  # Red
BORDER_COLOR = (0, 0, 0)  # Black
BUTTON_TEXT_COLOR = (0, 0, 0)  # Black
TEXT_COLOR = (0, 0, 0)  # Black
BG_COLOR = (255, 255, 255)  # White

# Initialize text output
current_word = []
full_text = []
frame_count = 0
flip_frame = False  # Toggle for frame flipping
predicted_gesture = "None"
max_confidence = 0.0

# Function to draw control panel (buttons and text)
def draw_control_panel(canvas):
    # Draw buttons
    for name, (x, y, w, h) in BUTTONS.items():
        color = EXIT_BUTTON_COLOR if name == "Exit" else BUTTON_COLOR
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)  # Fill
        cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 1)  # Black border
        cv2.putText(canvas, name, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BUTTON_TEXT_COLOR, 2)
    # Draw text with increased spacing
    cv2.putText(canvas, f"Gesture: {predicted_gesture} ({max_confidence:.2f})",
                (20, CAMERA_HEIGHT + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    cv2.putText(canvas, f"Word: {''.join(current_word)}",
                (20, CAMERA_HEIGHT + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    cv2.putText(canvas, f"Text: {''.join(full_text + [''.join(current_word)])}",
                (20, CAMERA_HEIGHT + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    cv2.putText(canvas, "Keys: D=Detect, SPACE=Space, BACK=Back, DEL=Clear, ENTER=Speak, f=Flip, q=Quit",
                (20, CAMERA_HEIGHT + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

# Mouse callback for button clicks
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for name, (bx, by, bw, bh) in BUTTONS.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                handle_button(name)

# Handle button actions
def handle_button(name):
    global current_word, full_text, running
    if name == "Detect" and predicted_gesture != "None":
        current_word.append(predicted_gesture)
        print(f"Letter added: {predicted_gesture}")
        save_text()
    elif name == "Space":
        if current_word:
            full_text.append("".join(current_word))
            current_word = []
            full_text.append(" ")
            print("Space added")
            save_text()
    elif name == "Backspace":
        if current_word:
            current_word.pop()
            print("Backspace: Removed last letter")
            save_text()
    elif name == "Clear":
        current_word = []
        full_text = []
        print("Cleared all text")
        save_text()
    elif name == "Speak":
        text_to_speak = "".join(full_text + ["".join(current_word)])
        if text_to_speak.strip():
            tts_engine.say(text_to_speak)
            tts_engine.runAndWait()
            print(f"Speaking: {text_to_speak}")
    elif name == "Exit":
        print("Exiting application")
        running = False

# Save text to file
def save_text():
    with open(TEXT_FILE, "w") as f:
        f.write("".join(full_text + ["".join(current_word)]))

# Open webcam and determine resolution
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREFERRED_CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREFERRED_CAMERA_HEIGHT)
if not cap.isOpened():
    print("Error: Could not open webcam. Trying index 1...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREFERRED_CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREFERRED_CAMERA_HEIGHT)
    if not cap.isOpened():
        print("Error: Could not open webcam with index 1. Falling back to 640x480...")
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FALLBACK_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FALLBACK_CAMERA_HEIGHT)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit(1)
        else:
            CAMERA_WIDTH, CAMERA_HEIGHT = FALLBACK_CAMERA_WIDTH, FALLBACK_CAMERA_HEIGHT
            print("Using fallback resolution: 640x480")
    else:
        CAMERA_WIDTH, CAMERA_HEIGHT = PREFERRED_CAMERA_WIDTH, PREFERRED_CAMERA_HEIGHT
        print("Using preferred resolution: 1280x720")
else:
    CAMERA_WIDTH, CAMERA_HEIGHT = PREFERRED_CAMERA_WIDTH, PREFERRED_CAMERA_HEIGHT
    print("Using preferred resolution: 1280x720")

# Define window dimensions
WINDOW_WIDTH = max(CAMERA_WIDTH, PANEL_WIDTH)
WINDOW_HEIGHT = CAMERA_HEIGHT + PANEL_HEIGHT

# Define button positions after webcam initialization
BUTTONS = {
    "Detect": (20, CAMERA_HEIGHT + 20, BUTTON_WIDTH, BUTTON_HEIGHT),
    "Space": (110, CAMERA_HEIGHT + 20, BUTTON_WIDTH, BUTTON_HEIGHT),
    "Backspace": (200, CAMERA_HEIGHT + 20, BUTTON_WIDTH, BUTTON_HEIGHT),
    "Clear": (290, CAMERA_HEIGHT + 20, BUTTON_WIDTH, BUTTON_HEIGHT),
    "Speak": (380, CAMERA_HEIGHT + 20, BUTTON_WIDTH, BUTTON_HEIGHT),
    "Exit": (470, CAMERA_HEIGHT + 20, BUTTON_WIDTH, BUTTON_HEIGHT)
}

# Create combined window
cv2.namedWindow("ASL Recognition")
cv2.setMouseCallback("ASL Recognition", mouse_callback)

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize frame to match CAMERA_WIDTH x CAMERA_HEIGHT
    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

    # Toggle frame flipping
    if flip_frame:
        frame = cv2.flip(frame, 1)

    # Skip every other frame to reduce CPU load
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        # Create combined canvas
        canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        canvas[:CAMERA_HEIGHT, :CAMERA_WIDTH] = frame
        canvas[CAMERA_HEIGHT:, :PANEL_WIDTH] = BG_COLOR  # White control panel
        # Draw buttons and text
        draw_control_panel(canvas)
        cv2.imshow("ASL Recognition", canvas)
        continue

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Process hand landmarks
    predicted_gesture = "None"
    max_confidence = 0.0
    if results.multi_hand_landmarks:
        print(f"Hand detected! Number of hands: {len(results.multi_hand_landmarks)}")
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks = scaler.transform(landmarks)

            predictions = model.predict(landmarks, verbose=0)
            confidence = np.max(predictions)
            class_idx = np.argmax(predictions)

            if confidence >= CONFIDENCE_THRESHOLD:
                if confidence > max_confidence:
                    predicted_gesture = index_to_class[class_idx]
                    max_confidence = confidence

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        print("No hand detected in this frame.")

    # Create combined canvas
    canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    canvas[:CAMERA_HEIGHT, :CAMERA_WIDTH] = frame
    canvas[CAMERA_HEIGHT:, :PANEL_WIDTH] = BG_COLOR  # White control panel

    # Draw buttons and text
    draw_control_panel(canvas)

    cv2.imshow("ASL Recognition", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Exiting application")
        running = False
    elif key == ord("d"):
        handle_button("Detect")
    elif key == ord(" "):
        handle_button("Space")
    elif key == 8:  # BACKSPACE
        handle_button("Backspace")
    elif key == 127:  # DELETE
        handle_button("Clear")
    elif key == 13:  # ENTER
        handle_button("Speak")
    elif key == ord("f"):
        flip_frame = not flip_frame
        print(f"Frame flipping: {flip_frame}")

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
tts_engine.stop()

# Save final text
save_text()
print(f"Recognized text: {''.join(full_text + [''.join(current_word)])}")
print(f"Text saved to {TEXT_FILE}")