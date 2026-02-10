import cv2
import mediapipe as mp
import pandas as pd
import os
import re

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create directory to store dataset
DATASET_DIR = "hand_landmarks_dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Constants
MAX_FRAMES = 1000  # Limit number of frames per gesture

def is_valid_gesture_name(name):
    """Validate gesture name: alphanumeric, no special chars, not empty."""
    return bool(name and re.match(r'^[a-zA-Z0-9_]+$', name))

def gesture_exists(name):
    """Check if gesture file already exists."""
    return os.path.exists(os.path.join(DATASET_DIR, f"{name}.csv"))

def collect_gesture():
    """Collect landmarks for a single gesture."""
    while True:
        gesture_name = input("Enter gesture name (or 'done' to finish): ").strip().lower()
        if gesture_name == 'done':
            return False
        if not is_valid_gesture_name(gesture_name):
            print("Invalid gesture name! Use alphanumeric characters and underscores only.")
            continue
        if gesture_exists(gesture_name):
            print(f"Gesture '{gesture_name}' already exists! Choose a different name.")
            continue
        break

    gesture_file = os.path.join(DATASET_DIR, f"{gesture_name}.csv")
    data = []
    frame_count = 0

    # Open webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return True

    # Lower confidence thresholds to improve detection
    with mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Debug: Check if hands are detected
            if results.multi_hand_landmarks:
                print(f"Hand detected! Number of hands: {len(results.multi_hand_landmarks)}")
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract X, Y, Z coordinates of all 21 hand landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])  # Storing X, Y, Z

                    # Save landmarks to dataset
                    data.append(landmarks)
                    frame_count += 1

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                print("No hand detected in this frame.")

            # Display frame
            cv2.putText(frame, f"Frames: {frame_count}/{MAX_FRAMES} | Press 's' to save, 'q' to quit",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Collect Hand Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and data:
                # Save data to CSV
                df = pd.DataFrame(data)
                df.to_csv(gesture_file, index=False, header=False)
                print(f"Gesture '{gesture_name}' saved successfully with {frame_count} frames!")
                break
            elif frame_count >= MAX_FRAMES:
                print(f"Reached maximum frame limit ({MAX_FRAMES}). Saving automatically.")
                df = pd.DataFrame(data)
                df.to_csv(gesture_file, index=False, header=False)
                print(f"Gesture '{gesture_name}' saved successfully with {frame_count} frames!")
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    """Main function to collect multiple gestures."""
    print("Start collecting gestures. Enter 'done' to finish.")
    while collect_gesture():
        pass
    print("Data collection complete.")

if __name__ == "__main__":
    main()