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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "own_income",
    "job_tenure",
    "rating",
    "risk_score",
    "household_income",
    "customer_duration",
    "customer_satisfaction",
    "sex"
]
label_column = "churn_risk"

feature_columns_dtype = {
    "own_income": np.float64,
    "job_tenure": np.float64,
    "rating": np.float64,
    "risk_score": np.float64,
    "household_income": np.float64,
    "customer_duration": np.float64,
    "customer_satisfaction": np.float64,
    "sex": str,
}
label_column_dtype = {"churn_risk": np.float64}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z

logger.debug("Starting preprocessing.")
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", dest="input_data", type=str, required=True)
args = parser.parse_args()

input_data = args.input_data

base_dir = "."
pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
#input_data = args.input_data
bucket_name = input_data.split("/")[2]
key = "/".join(input_data.split("/")[3:])

logger.info("Downloading data from bucket: %s, key: %s", bucket_name, key)
fn = f"{base_dir}/data/customerchurn.csv"
#s3 = boto3.resource("s3")
#s3.Bucket(bucket).download_file(key, fn)
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(key)
blob.download_to_filename(fn)

logger.debug("Reading downloaded data.")
df = pd.read_csv(
    fn,
    header=None,
    names=feature_columns_names + [label_column],
    dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
)
os.unlink(fn)

logger.debug("Defining transformers.")
numeric_features = list(feature_columns_names)
numeric_features.remove("sex")
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["sex"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

logger.info("Applying transforms.")
y = df.pop("churn_risk")
X_pre = preprocess.fit_transform(df)
y_pre = y.to_numpy().reshape(len(y), 1)

X = np.concatenate((y_pre, X_pre), axis=1)

logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
np.random.shuffle(X)
train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

output_rel_paths = {
    'train': 'train/train.csv',
    'validation': 'validation/validation.csv',
    'test': 'test/test.csv'
}

for k in output_rel_paths.keys():
    os.mkdir(f"{base_dir}/{k}")

output_bkt_rel_paths = {k:f"processed_data/{v}" for k,v in output_rel_paths.items()}
output_local_abs_paths = {k:f"{base_dir}/{v}" for k,v in output_rel_paths.items()}

logger.info("Writing out datasets to %s.", output_bkt_rel_paths)
pd.DataFrame(train).to_csv(output_local_abs_paths['train'], header=False, index=False)
pd.DataFrame(validation).to_csv(
    output_local_abs_paths['validation'], header=False, index=False
)
pd.DataFrame(test).to_csv(output_local_abs_paths['test'], header=False, index=False)

train_blob = bucket.blob(output_bkt_rel_paths['train'])
validation_blob = bucket.blob(output_bkt_rel_paths['validation'])
test_blob = bucket.blob(output_bkt_rel_paths['test'])

train_blob.upload_from_filename(output_local_abs_paths['train'])
validation_blob.upload_from_filename(output_local_abs_paths['validation'])
test_blob.upload_from_filename(output_local_abs_paths['test'])

# return (f"gs://{bucket_name}/{output_bkt_rel_paths['train']}", f"gs://{bucket_name}/{output_bkt_rel_paths['validation']}", f"gs://{bucket_name}/{output_bkt_rel_paths['test']}")