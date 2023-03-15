# Module imports
import logging
import os
import pathlib
import requests
import tempfile
import argparse

#import boto3
from google.cloud import storage
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import hypertune
import joblib
#from sklearn.externals import joblib

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Hyperparameter inputs

parser = argparse.ArgumentParser()

parser.add_argument(
    '--learning_rate',
    required=True,
    type=float,
    help='learning rate')

parser.add_argument(
    '--subsample',
    required=True,
    type=float,
    help='Subsample')

parser.add_argument(
    '--training-dataset-path',
    dest='training_dataset_path',
    required=True,
    type=str,
    help="GCS path to the training dataset (csv)")

parser.add_argument(
    '--test-dataset-path',
    dest='test_dataset_path',
    required=True,
    type=str,
    help='GCS path to the test dataset (csv)')

parser.add_argument(
    '--model-dir',
    dest='model_dir',
    default=os.getenv('AIP_MODEL_DIR'),
    type=str, help='Model dir.')

args = parser.parse_args()

# End Hyperparameter inputs

base_dir = "."
pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
NUM_EPOCHS = 10
training_data = f"{base_dir}/data/train.csv"
test_data = f"{base_dir}/data/test.csv"

storage_client = storage.Client()

training_dataset_path = args.training_dataset_path
test_dataset_path = args.test_dataset_path

bucket_name = training_dataset_path.split("/")[2]
training_key = "/".join(training_dataset_path.split("/")[3:])
test_key = "/".join(test_dataset_path.split("/")[3:])
#test_key = "/".join(test_dataset_path.split("/")[3:])

bucket = storage_client.bucket(bucket_name)
training_blob = bucket.blob(training_key)
training_blob.download_to_filename(training_data)

test_blob = bucket.blob(test_key)
test_blob.download_to_filename(test_data)

#test_blob = bucket.blob(test_key)
#test_blob.download_to_filename(test_data)

# Since we get a headerless CSV file we specify the column names here.

def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z

logger.debug("Reading downloaded data.")
train_df = pd.read_csv(
    training_data,
    header=None
)
os.unlink(training_data)

test_df = pd.read_csv(
    test_data,
    header=None
)
os.unlink(test_data)

y_train = train_df.iloc[:, 0].to_numpy()
train_df.drop(train_df.columns[0], axis=1, inplace=True)
X_train = train_df

y_test = test_df.iloc[:, 0].to_numpy()
test_df.drop(test_df.columns[0], axis=1, inplace=True)
X_test = test_df

reg = GradientBoostingRegressor(learning_rate=args.learning_rate, subsample=args.subsample)
reg.fit(X_train, y_train)

predictions = reg.predict(X_test)


mse = mean_squared_error(y_test, predictions)

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='mse',
    metric_value=mse,
    global_step=NUM_EPOCHS)

joblib.dump(reg, f"{base_dir}/model.joblib")

output_dest = args.model_dir

logger.info(f"Attempting to upload model to {output_dest}")

output_bucket_name = output_dest.split("/")[2]
output_key = "/".join(output_dest.split("/")[3:])

output_bucket = storage_client.bucket(output_bucket_name)

model_blob = bucket.blob(f"{output_key}/model.joblib")
model_blob.upload_from_filename(f"{base_dir}/model.joblib")