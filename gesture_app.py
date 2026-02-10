import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)

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
frame_count = 0
flip_frame = False  # Toggle for frame flipping

# Open webcam with explicit settings
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Could not open webcam. Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam with index 1.")
        exit(1)

# Initialize text output
recognized_text = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Debug: Log frame dimensions
    print(f"Frame dimensions: {frame.shape}")

    # Toggle frame flipping with 'f' key
    if flip_frame:
        frame = cv2.flip(frame, 1)

    # Skip every other frame to reduce CPU load
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("ASL Recognition", frame)
        continue

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_gesture = "None"
    max_confidence = 0.0

    # Debug: Check if hands are detected
    if results.multi_hand_landmarks:
        print(f"Hand detected! Number of hands: {len(results.multi_hand_landmarks)}")
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract X, Y, Z coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Normalize landmarks
            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks = scaler.transform(landmarks)

            # Predict gesture
            predictions = model.predict(landmarks, verbose=0)
            confidence = np.max(predictions)
            class_idx = np.argmax(predictions)

            if confidence >= CONFIDENCE_THRESHOLD:
                if confidence > max_confidence:
                    predicted_gesture = index_to_class[class_idx]
                    max_confidence = confidence

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        print("No hand detected in this frame.")
        cv2.putText(frame, "No Hand Detected", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display prediction and confidence
    cv2.putText(frame, f"Gesture: {predicted_gesture} ({max_confidence:.2f})",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 's' to save, 'f' to flip, 'q' to quit", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("ASL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s") and predicted_gesture != "None":
        recognized_text.append(predicted_gesture)
        with open(TEXT_FILE, "a") as f:
            f.write(predicted_gesture + "\n")
        print(f"Saved gesture: {predicted_gesture}")
    elif key == ord("f"):
        flip_frame = not flip_frame
        print(f"Frame flipping: {flip_frame}")

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

# Print and save final text
final_text = "".join(recognized_text)
print(f"Recognized text: {final_text}")
with open(TEXT_FILE, "a") as f:
    f.write("\nFinal text: " + final_text + "\n")
print(f"Text saved to {TEXT_FILE}")