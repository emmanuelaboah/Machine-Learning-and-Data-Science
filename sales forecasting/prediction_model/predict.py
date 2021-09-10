# Import relevant libraries
from preprocess_data import load_data
from keras.models import load_model
import pandas as pd

# Load the test dataset
X_test = load_data("sales_data_testing_scaled.csv")

# Load the trained mode into the workspace
model = load_model("h5_file/model.h5")

# Make predictions
prediction = model.predict(X_test)

# Grab just the first element of the first prediction (since we only have one)
prediction = prediction[0][0]

print("Earnings Prediction for Proposed Product - ${}".format(prediction))


