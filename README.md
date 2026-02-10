Genetic Algorithm Enhanced CNN for Robust Interpretation of ASL
A computer vision-based application that translates American Sign Language (ASL) hand gestures into text and spoken speech in real-time. This project leverages MediaPipe for landmark extraction and a Deep Neural Network (DNN) optimized via Genetic Algorithms for high-accuracy classification.

📖 Table of Contents
Project Overview

Key Features

Tech Stack

Project Architecture

Installation

How to Run (Step-by-Step)

Model & Optimization

Future Improvements

🚀 Project Overview
Communication barriers between the Deaf/Hard-of-Hearing community and the general public can be significant. This project aims to bridge that gap by creating a lightweight, real-time translator that requires only a standard webcam.

Unlike image-based classifiers (CNNs) that are computationally heavy, this project uses geometric landmark data (21 points per hand), making it fast enough to run on standard CPUs without a GPU.

✨ Key Features
Real-Time Detection: Instant feedback using MediaPipe's efficient hand tracking.

Speech Synthesis: Converts recognized text strings into audio using pyttsx3.

Custom Dataset Generation: Built-in tools to record and label new gestures easily.

Automated Optimization: Uses Genetic Algorithms (DEAP) to automatically find the best Neural Network architecture (hyperparameter tuning).

Interactive GUI: On-screen controls for sentence construction, deletion, and audio playback.

🛠 Tech Stack
Language: Python

Computer Vision: OpenCV, MediaPipe

Machine Learning: TensorFlow / Keras, Scikit-Learn

Optimization: DEAP (Distributed Evolutionary Algorithms in Python)

Data Processing: NumPy, Pandas

Visualization: Matplotlib, Seaborn

📂 Project Architecture
The repository is structured to separate data collection, processing, and model logic:

Plaintext
├── collect_landmarks.py    # Captures webcam frames & saves landmarks to CSV
├── preprocess_data.py      # Normalizes data, augments noise, & splits Train/Test
├── inspect_data.py         # Validates dataset integrity, shapes, and feature counts
├── train_model.py          # Trains the baseline Keras Neural Network
├── optimize_model.py       # Runs Genetic Algorithm to find the best model architecture
├── gesturetest.py          # Main Application: Real-time inference & GUI
├── gui.py                  # (Optional) Standalone GUI interface wrapper
├── run.bat                 # Batch script for one-click launch
└── requirements.txt        # Python dependencies
📦 Installation
Clone the Repository

Bash
git clone https://github.com/yourusername/SignLanguageAI.git
cd SignLanguageAI
Create a Virtual Environment (Recommended)

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

Bash
pip install -r requirements.txt
(Note: Ensure you have opencv-python, mediapipe, tensorflow, deap, and pyttsx3 installed).

🚦 How to Run (Step-by-Step)
This project follows a reproducible ML pipeline. You can either use the pre-trained models or train your own from scratch.

Phase 1: Data Collection
If you want to create your own gestures:

Bash
python collect_landmarks.py
Enter a gesture name (e.g., "Hello").

The script captures 1000 frames of hand landmarks.

Data is saved to hand_landmarks_dataset/.

Phase 2: Preprocessing
Clean, normalize, and augment the data:

Bash
python preprocess_data.py
Generates X_train.npy, y_train.npy, etc.

Creates scaler.pkl (for normalization) and class_indices.pkl.

Optional: Verify Data If you want to ensure your data shapes are correct before training:

Bash
python inspect_data.py
Checks if input features = 63 (21 landmarks × 3 coords).

Confirms sample counts match between X and y.

Phase 3: Training
Option A: Basic Training

Bash
python train_model.py
Trains a standard model and saves it as hand_gesture_model.h5.

Generates confusion matrices and loss plots.

Option B: Evolutionary Optimization (Advanced)

Bash
python optimize_model.py
Runs a Genetic Algorithm to evolve the optimal number of neurons, layers, and dropout rates.

Saves the best performer as hand_gesture_model_optimized.h5.

Phase 4: Real-Time Testing
Run the main application:

Bash
python gesturetest.py
Controls:

D: Detect current gesture and add to sentence.

SPACE: Add space.

ENTER: Speak the sentence (Text-to-Speech).

BACKSPACE: Remove last character.

🧠 Model & Optimization
Input Features
The model does not take raw images as input. Instead, it uses the (x, y, z) coordinates of 21 hand landmarks extracted by MediaPipe.

Input Shape: 63 features (21 points × 3 coords).

Benefits: Invariant to lighting conditions and background noise; extremely lightweight.

Genetic Algorithm (GA)
To ensure the highest accuracy, we implemented an evolutionary strategy using DEAP to tune hyperparameters:

Genes: Number of Neurons, Dropout Rate, Learning Rate, Number of Layers.

Fitness Function: Validation Accuracy.

Selection: Tournament Selection.

Mutation: Gaussian Mutation.

This automated search allows the model to adapt specifically to the complexity of the dataset provided.

🔮 Future Improvements
LSTM Integration: Upgrade the model to support dynamic gestures (movement over time) rather than just static poses.

Two-Hand Support: Expand the input vector to handle two-handed ASL signs.

Mobile App: Convert the model to TensorFlow Lite for deployment on Android/iOS.

👥 Credits
MediaPipe by Google for the robust hand tracking solution.

DEAP Library for the evolutionary computation framework.

Developed by Nikhil Rawat and Naitik Pundir as a Final Year Project.
