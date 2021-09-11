
# Import relevant libraries
import tensorflow as tf
if tf.executing_eagerly():  # use tensorflow compatibility and disable eager mode
    tf.compat.v1.disable_eager_execution()  # for running the export model for tensorflow
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
    epochs=120,
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


# Export the model into a tensorflow format that is suitable for deployment into the google cloud's ML engine
model_builder = tf.compat.v1.saved_model.builder.SavedModelBuilder("export_model")

# Declare the inputs of the model builder
inputs = {
    "input": tf.compat.v1.saved_model.utils.build_tensor_info(model.input)
}

# Declare the outputs of the model builder
outputs = {
    "output": tf.compat.v1.saved_model.utils.build_tensor_info(model.output)
}

# tensorflow signature definition
def_signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
)

# save the structure and the training weights of the model builder
model_builder.add_meta_graph_and_variables(
    tf.compat.v1.keras.backend.get_session(),
    tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
    signature_def_map={tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: def_signature}
)

# save the model
model_builder.save()

