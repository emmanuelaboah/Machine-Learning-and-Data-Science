# End-to-End Deep Learning Project: Video Game Earnings Forecasting

This is an end-to-end Supervised Machine Learning Project for predicting how much money to expect future video game
to earn based on historical data. The process involved preprocessing of the video game data to the deployment of
model into production using Google Cloud ML-service.

**Objective**: Develop a regression model using deep learning for predicting the total 
earnings of a video game. 

**Tools and Packages**: Tensorflow, Keras, Cloud Computing, Scikit-learn, Tensorboard, Google SDK,
Google Cloud ML-Service, JSON.

**The full ML pipeline or steps involved in this end-to-end project are:**
 - Data preprocessing
 - Deep learning model building and training using the sales data
 - Visualization of model architecture and training progress using Tensorboard
 - Deployment of model into production on Google cloud ML-Service


**Dataset**:

The datasets can be found in the <i class="icon-cog"></i> **[datasets](https://github.com/emmanuelaboah/Machine-Learning-and-Data-Science/tree/master/sales%20forecasting/datasets)**
directory. They consist of the raw historical data and the preprocessed data.

Reference to the raw data: *Adam Geitgey*

**Data Preprocessing**:

The Python Script for preprocessing the raw data can be found in the 
<i class="icon-cog"></i> **[dataset_preprocessing](https://github.com/emmanuelaboah/Machine-Learning-and-Data-Science/tree/master/sales%20forecasting/dataset_preprocessing)**
directory. It requires a path to the dataset.

### Deep Learning Model:

The details of the implementation of the deep learning model can be found in 
the <i class="icon-cog"></i> **[prediction_model](https://github.com/emmanuelaboah/Machine-Learning-and-Data-Science/blob/master/sales%20forecasting/prediction_model/model.py)**
directory. The error rate (mean squared error) of the model on the test data was
below **0.40**.

Tensorboard logger was also created for visualization of the model 
architecture and the training progress.

The *predict.py* script can also be used for prediction on a local machine.

### Model Deployment
The trained model was exported to a protobuff format (*.pb*) and hosted 
on Google cloud service for online prediction.
The details of requesting for prediction from the deployed model in 
the cloud can be found in the **[request_prediction_from_cloud.py](https://github.com/emmanuelaboah/Machine-Learning-and-Data-Science/blob/master/sales%20forecasting/prediction_model/request_prediction_from_cloud.py)**
script.

