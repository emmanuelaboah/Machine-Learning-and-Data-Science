# Import relevant libraries
import googleapiclient.discovery
from oauth2client.client import GoogleCredentials

# Assumption: Google cloud platform for machine learning has already been created
# Modify the input parameters below accordingly
# Copy info from your google AI-Platform into the ff:
PROJECT_ID = "name of your project id from google cloud"
MODEL_NAME = "name of how you saved your mode in the cloud"
CREDENTIALS_FILE = "google_cloud_credentials.json"  # this file can be obtained from your  google cloud account

# sample input sales data for prediction
# The input has already been scaled for prediction purposes
inputs_for_prediction = [
    {"input": [0.666, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.48]}
]

# Use the credentials file copied from Google cloud to
# connect to the Google Cloud-ML Service
gcloud_credentials = GoogleCredentials.from_stream(CREDENTIALS_FILE)
service = googleapiclient.discovery.build('ml', 'v1', credentials=gcloud_credentials)

# Connect to the Prediction Model in Google cloud service
name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
response = service.projects().predict(
    name=name,
    body={'instances': inputs_for_prediction}
).execute()

# Raise and error in case of any runtime errors
if 'error' in response:
    raise RuntimeError(response['error'])

# Grab the output from the response object
output = response['predictions']

# Display the output
print(output)
