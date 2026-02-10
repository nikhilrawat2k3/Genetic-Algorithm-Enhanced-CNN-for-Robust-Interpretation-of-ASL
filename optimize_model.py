import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from deap import base, creator, tools, algorithms
import random
import pickle

# Load class indices to verify gesture names
try:
    with open("class_indices.pkl", "rb") as f:
        class_indices = pickle.load(f)
    print("Gesture names:", class_indices)
except FileNotFoundError:
    print("Warning: class_indices.pkl not found!")

# Load preprocessed data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Create validation set from training data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Verify data shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"y_test shape: {y_test.shape}")

# Define fitness function for GA
def evaluate(individual):
    try:
        num_neurons = int(individual[0])
        dropout_rate = individual[1]
        learning_rate = individual[2]
        num_layers = int(individual[3])

        # Build model
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        model.add(Dense(num_neurons, activation="relu"))
        model.add(Dropout(dropout_rate))

        # Add additional hidden layers
        for _ in range(num_layers - 1):
            model.add(Dense(num_neurons // 2, activation="relu"))
            model.add(Dropout(dropout_rate))

        model.add(Dense(len(np.unique(y_train)), activation="softmax"))

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Define early stopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=128,
            callbacks=[early_stopping],
            verbose=0
        )

        # Get validation accuracy
        val_accuracy = history.history["val_accuracy"][-1]
        return (val_accuracy,)

    except Exception as e:
        print(f"Error evaluating individual {individual}: {e}")
        return (0.0,)  # Return low fitness for failed models

# GA setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("num_neurons", random.randint, 64, 256)
toolbox.register("dropout_rate", random.uniform, 0.2, 0.5)
toolbox.register("learning_rate", random.uniform, 0.0001, 0.01)
toolbox.register("num_layers", random.randint, 1, 3)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.num_neurons, toolbox.dropout_rate, toolbox.learning_rate, toolbox.num_layers), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
population = toolbox.population(n=20)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

print("Starting genetic algorithm optimization...")
hof = tools.HallOfFame(1)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)

# Get best individual
best_individual = hof[0]
num_neurons = int(best_individual[0])
dropout_rate = best_individual[1]
learning_rate = best_individual[2]
num_layers = max(1, int(best_individual[3]))
print(f"\nBest hyperparameters: Neurons={num_neurons}, Dropout={dropout_rate:.4f}, Learning Rate={learning_rate:.6f}, Layers={num_layers}")

# Train final optimized model
optimized_model = Sequential()
optimized_model.add(Input(shape=(X_train.shape[1],)))
optimized_model.add(Dense(num_neurons, activation="relu"))
optimized_model.add(Dropout(dropout_rate))
for _ in range(num_layers - 1):
    optimized_model.add(Dense(num_neurons // 2, activation="relu"))
    optimized_model.add(Dropout(dropout_rate))
optimized_model.add(Dense(len(np.unique(y_train)), activation="softmax"))

# Print model summary
print("\nFinal model summary:")
optimized_model.summary()

optimized_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = optimized_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,  # Reduced for faster testing
    batch_size=128,
    callbacks=[early_stopping]
)

# Evaluate on test set
test_loss, test_accuracy = optimized_model.evaluate(X_test, y_test)
print(f"\nFinal Model Test Results:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")

# Save the optimized model
optimized_model.save("hand_gesture_model_optimized.h5")
print("✅ Optimized model saved as 'hand_gesture_model_optimized.h5'")