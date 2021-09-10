
# Import relevant libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from preprocess_data import load_data, save_plot, save_data
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pandas as pd

RUN_NAME = "run 2 with 4 layers"

# Load the training data
X_train = load_data("sales_data_training_scaled.csv")
y_train = load_data("y_train.csv")

# Load the testing data
X_test = load_data("sales_data_testing_scaled.csv")
y_test = load_data("y_test.csv")

# Specify early stopping to check overfitting
es = EarlyStopping(monitor="loss", patience=10)

# Define the model
model = keras.Sequential([
    layers.Dense(150, activation ="relu", name="layer1", input_dim=9),
    layers.Dense(64, activation="relu", name="layer2"),
    layers.Dense(32, activation="relu", name="layer3"),
    layers.Dense(1, name="output_layer")
]
)

# compile the model
model.compile(loss="mse", optimizer="adam")

# Display a summary of the model
print(model.summary())

# Create a Tensorboard logger
logger = TensorBoard(
    log_dir="logs/{}".format(RUN_NAME),  # directory where to save the log files
    write_graph=True,  # visualize the graph of the NN in TensorBoard
    histogram_freq=5,  # freq. (in epochs) at which to compute act. and weight hist. for the layers of the model
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    # validation_split=0.1,
    epochs=100,
    shuffle=True,
    verbose=2,
    # callbacks=[es]
    callbacks=[logger]
)

# Evaluate the model on the test set
test_error = model.evaluate(X_test, y_test, verbose=0)
print("The mean squared error for the test data is: {:.3f}".format(test_error))

# Save the neural network model
# model.save("h5_file/model.h5")
print("Model is saved to disk.")
