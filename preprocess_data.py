import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Directory containing gesture data
DATASET_DIR = "hand_landmarks_dataset"

# Constants
NOISE_STD = 0.01  # Standard deviation for noise augmentation

def validate_csv(file_path):
    """Validate CSV file: non-empty and correct number of columns."""
    try:
        df = pd.read_csv(file_path, header=None)
        if df.empty:
            return False
        # Expect 63 columns (21 landmarks x 3 coordinates: X, Y, Z)
        if df.shape[1] != 63:
            print(f"Warning: {file_path} has {df.shape[1]} columns, expected 63.")
            return False
        return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def augment_data(data):
    """Add Gaussian noise to landmark data for augmentation."""
    noise = np.random.normal(0, NOISE_STD, data.shape)
    return data + noise

def main():
    # Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory '{DATASET_DIR}' not found!")

    # Load and validate gesture files
    gestures = []
    labels = []
    gesture_mapping = {}  # Map gesture names to indices
    gesture_index = 0

    for file in sorted(os.listdir(DATASET_DIR)):
        if file.endswith(".csv"):
            file_path = os.path.join(DATASET_DIR, file)
            if not validate_csv(file_path):
                print(f"Skipping invalid file: {file}")
                continue

            # Load CSV
            df = pd.read_csv(file_path, header=None)
            gesture_name = file[:-4]  # Remove .csv extension
            gesture_mapping[gesture_name] = gesture_index

            # Append original and augmented data
            gestures.append(df.values)
            gestures.append(augment_data(df.values))  # Add noisy version
            labels.append(np.full((df.shape[0],), gesture_index))  # Original labels
            labels.append(np.full((df.shape[0],), gesture_index))  # Augmented labels
            gesture_index += 1

    if not gestures:
        raise ValueError("No valid gesture files found in the dataset directory!")

    # Convert to NumPy arrays
    X = np.vstack(gestures)  # Combine all gesture data
    y = np.hstack(labels)    # Combine all labels

    # Normalize data using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save preprocessed data
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    # Save scaler for inference
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save gesture mapping (same as class_indices.pkl for consistency)
    with open("class_indices.pkl", "wb") as f:
        pickle.dump(gesture_mapping, f)

    print(f"✅ Preprocessing complete! Processed {len(gesture_mapping)} gestures.")
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    print("Data saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    print("Scaler saved: scaler.pkl")
    print("Gesture mapping saved: class_indices.pkl")

if __name__ == "__main__":
    main()