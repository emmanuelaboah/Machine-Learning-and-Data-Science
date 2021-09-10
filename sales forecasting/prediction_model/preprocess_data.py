# Import relevant libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import os


# Import datasets
def load_data(data, path="datasets"):
    """
    Function for loading datasets in csv format

        :arg:
            1. data: name of dataset
            2. path (optional): path to load the dataset
    """

    print("Loading dataset to workspace..")
    data_path = os.path.join(".", path, data)
    print("Done!")
    return pd.read_csv(data_path)


# Saving plots and figures
def save_plot(fig, path="plots", tight_layout=True, fig_extension="png", resolution=300):
    """
        Function for saving figures and plots

        :arg
            1. fig: label of the figure
            2. path (optional): output path of the figure
    """

    fig_path = os.path.join(".", "images", path, fig + "." + fig_extension)

    print("Saving figure...", fig)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)
    print("figure can be found in: ", path)


# Pipeline for scaling the training and test data
def data_scaling(train, test):
    """"
        Function for scaling the training and test dataset
        to a range of 0 and 1

        :arg
            1. train: train dataset
            2. test: test dataset
    """

    print("fitting and transforming the train data...")
    pipe = Pipeline(steps=[("scaler", MinMaxScaler(feature_range=(0, 1)))])
    scaled_train = pipe.fit_transform(train)
    scaled_train = pd.DataFrame(scaled_train, columns=train.columns.values)
    print("Done!")

    print("transforming the testing dataset...")
    scaled_test = pipe.transform(test)
    scaled_test = pd.DataFrame(scaled_test, columns=test.columns.values)
    print("Done!")

    return scaled_train, scaled_test


# Save the scaled data into a csv format in a specified directory
def save_data(data, label, path="datasets"):
    """
    Function for saving the scaled data to csv format

    :param data: dataset
    :param label: name of the output dataset (for saving)
    :param path: directory for saving the output
    :return: saved dataset
    """

    path = os.path.join(".", path)
    print("Dataset saved!")
    return data.to_csv(os.path.join(".", path, label), index=False)


# Load training data set from CSV file
training_data = load_data("sales_data_training.csv")

# Load testing data set from CSV file
test_data = load_data("sales_data_test.csv")

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
# Scale both the training inputs and outputs
scaled_data = data_scaling(training_data.iloc[:, : -1], test_data.iloc[:, : -1])
scaled_training = scaled_data[0]
scaled_testing = scaled_data[1]

# Save scaled data dataframes to new CSV files
scaled_training_df = save_data(scaled_training, "sales_data_training_scaled.csv")
scaled_testing_df = save_data(scaled_testing, "sales_data_testing_scaled.csv")

# save the train and test targets
y_train = save_data(training_data["unit_price"], "y_train.csv")
y_test = save_data(test_data["unit_price"], "y_test.csv")