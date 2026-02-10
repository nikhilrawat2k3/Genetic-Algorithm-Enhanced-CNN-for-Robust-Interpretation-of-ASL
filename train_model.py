import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Clear Keras session to avoid cached states
K.clear_session()

# Load preprocessed data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Verify data shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Use a subset of the data for debugging
subset_size = 1000
X_train_subset = X_train[:subset_size]
y_train_subset = y_train[:subset_size]
X_test_subset = X_test[:subset_size//4]
y_test_subset = y_test[:subset_size//4]
print(f"Training on subset: X_train_subset shape: {X_train_subset.shape}, y_train_subset shape: {y_train_subset.shape}")

# Get number of unique gestures (classes)
num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")

# Build a minimal neural network model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(num_classes, activation="softmax")
])

# Print model summary
model.summary()

# Compile the model with basic metrics
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Test model output shape
sample_input = X_train_subset[:2]
sample_output = model.predict(sample_input)
print(f"Sample prediction shape: {sample_output.shape}")

# Define callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train the model on subset
history = model.fit(
    X_train_subset,
    y_train_subset,
    validation_data=(X_test_subset, y_test_subset),
    epochs=20,
    batch_size=128,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_subset, y_test_subset)
print(f"\nTest Results:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")

# Generate confusion matrix
y_pred = np.argmax(model.predict(X_test_subset), axis=1)
cm = confusion_matrix(y_test_subset, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_subset, y_pred))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_history.png")
plt.close()

# Save the trained model
model.save("hand_gesture_model.h5")
print("✅ Model training complete! Saved as 'hand_gesture_model.h5'")
print("Confusion matrix saved as 'confusion_matrix.png'")
print("Training history plot saved as 'training_history.png'")