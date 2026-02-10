import numpy as np

def inspect_data():
    try:
        # Load preprocessed data
        X_train = np.load("X_train.npy")
        X_test = np.load("X_test.npy")
        y_train = np.load("y_train.npy")
        y_test = np.load("y_test.npy")

        # Print shapes
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Check expected features
        expected_features = 63  # 21 landmarks x 3 coordinates (X, Y, Z)
        if X_train.shape[1] != expected_features:
            print(f"Error: X_train has {X_train.shape[1]} features, expected {expected_features}")
        else:
            print("X_train feature count is correct.")

        # Check sample counts
        if X_train.shape[0] != y_train.shape[0]:
            print(f"Error: X_train has {X_train.shape[0]} samples, but y_train has {y_train.shape[0]}")
        else:
            print("X_train and y_train sample counts match.")

        if X_test.shape[0] != y_test.shape[0]:
            print(f"Error: X_test has {X_test.shape[0]} samples, but y_test has {y_test.shape[0]}")
        else:
            print("X_test and y_test sample counts match.")

        # Check number of classes
        num_classes = len(np.unique(y_train))
        print(f"Number of classes: {num_classes}")

        # Check a sample of the data
        print("\nSample X_train data (first row):")
        print(X_train[0])
        print("\nSample y_train data (first 10 labels):")
        print(y_train[:10])

    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    inspect_data()