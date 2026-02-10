import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import mediapipe as mp
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import pickle

# Load the trained model
model = tf.keras.models.load_model("hand_gesture_model_optimized.h5")

# Load class indices
with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)
reverse_class_indices = {v: k for k, v in class_indices.items()}  # Reverse mapping

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()

# Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Global variables
detected_text = ""
cap = cv2.VideoCapture(0)  # Open the camera

# Function to process video and update GUI
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert to ImageTk format
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    
    camera_label.after(10, update_frame)  # Continuously update frame

# Function to detect and add letter
def detect_gesture():
    global detected_text
    ret, frame = cap.read()
    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark points
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            
            # Convert landmarks into numpy array
            landmarks = np.array(landmarks).reshape(1, -1)
            
            # Predict gesture
            prediction = model.predict(landmarks)
            predicted_class = np.argmax(prediction)
            detected_letter = reverse_class_indices[predicted_class]  # Get the detected letter
            
            # Append detected letter to text
            detected_text += detected_letter
            text_label.config(text=f"Detected Text: {detected_text}")

# Function to delete last letter (Backspace)
def backspace():
    global detected_text
    detected_text = detected_text[:-1]  # Remove last character
    text_label.config(text=f"Detected Text: {detected_text}")

# Function to add a space
def add_space():
    global detected_text
    detected_text += " "  # Add a space
    text_label.config(text=f"Detected Text: {detected_text}")

# Function to speak the detected text
def speak_text():
    tts_engine.say(detected_text)
    tts_engine.runAndWait()

# Function to clear the text field
def clear_text():
    global detected_text
    detected_text = ""
    text_label.config(text="Detected Text: ")

# GUI Setup
app = tk.Tk()
app.title("Sign Language to Speech")
app.geometry("600x550")

# Camera feed label
camera_label = Label(app)
camera_label.pack()

# Text display label
text_label = Label(app, text="Detected Text: ", font=("Arial", 14))
text_label.pack(pady=10)

# Buttons
detect_btn = Button(app, text="Detect", command=detect_gesture, font=("Arial", 12), bg="lightblue")
detect_btn.pack(pady=5)

space_btn = Button(app, text="Space", command=add_space, font=("Arial", 12), bg="lightgray")
space_btn.pack(pady=5)

backspace_btn = Button(app, text="Backspace", command=backspace, font=("Arial", 12), bg="orange")
backspace_btn.pack(pady=5)

speak_btn = Button(app, text="Speak", command=speak_text, font=("Arial", 12), bg="lightgreen")
speak_btn.pack(pady=5)

clear_btn = Button(app, text="Clear", command=clear_text, font=("Arial", 12), bg="yellow")
clear_btn.pack(pady=5)

exit_btn = Button(app, text="Exit", command=app.quit, font=("Arial", 12), bg="red")
exit_btn.pack(pady=5)

# Start updating the camera feed
update_frame()

app.mainloop()

# Release camera when app is closed
cap.release()
cv2.destroyAllWindows()
