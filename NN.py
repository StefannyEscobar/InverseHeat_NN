import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np

# Load the dataset
data = np.load('heat_transfer_data.npz', allow_pickle=True)
dataset = data['dataset']

# Prepare input and output data
X = np.array([entry['params'] for entry in dataset])
y = np.array([entry['temperature_distribution'] for entry in dataset])

# Define the neural network
model = Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(y[0]))  # Adjust based on the size of your temperature distribution
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network
history = model.fit(X, y, epochs=50, validation_split=0.2)

# Save the trained model in the native Keras format
model.save('trained_model')  # Note that you don't need to include the file extension

# Plot the training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
