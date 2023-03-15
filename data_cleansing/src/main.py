# Azure compatible code with comments where changing

import logging
import os
import pathlib
import requests
import tempfile
import argparse
import azure.storage.blob
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(handler)

# Define constants
FEATURE_COLUMNS_NAMES = [
    "own_income",
    "job_tenure",
    "rating",
    "risk_score",
    "household_income",
    "customer_duration",
    "customer_satisfaction",
    "sex"
]
LABEL_COLUMN = "churn_risk"
FEATURE_COLUMNS_DTYPE = {
    "own_income": np.float64,
    "job_tenure": np.float64,
    "rating": np.float64,
    "risk_score": np.float64,
    "household_income": np.float64,
    "customer_duration": np.float64,
    "customer_satisfaction": np.float64,
    "sex": str,
}
LABEL_COLUMN_DTYPE = {"churn_risk": np.float64}

# Define function to merge two dictionaries


def merge_two_dicts(dict1, dict2):
    """Merge two dictionaries, returning a new copy. If duplicate keys, values of the second dictionary will override the first."""
    return {**dict1, **dict2}


if __name__ == '__main__':

    logger.debug("Starting preprocessing")

    # Parse arguments

    parser = argparse.ArgumentParser(description='Process strings.')
    parser.add_argument('--input-data', dest='input_data', type=str, required=True,
                        help='an input data for processing')
    args = parser.parse_args()

    input_data = args.input_data

    # Create directories to store data
    base_dir = "."
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    for k in ['train', 'validation', 'test']:
        os.mkdir(f"{base_dir}/{k}")

    # Download data from Azure Blob Storage
    logger.info("Downloading data from Azure Blob Storage")
    _, filename = tempfile.mkstemp()
    blob_service_client = azure.storage.blob.BlobServiceClient.from_connection_string(
        input_data)
    container_client = blob_service_client.get_container_client(
        'azureml')
    blob_client = container_client.get_blob_client('customerchurn.csv')
    with open(filename, "wb") as my_blob:
        download_stream = blob_client.download_blob()
        my_blob.write(download_stream.readall())

    # Read data into Pandas DataFrame
    logger.debug("Reading downloaded data into Pandas DataFrame")
    df = pd.read_csv(
        filename,
        header=None,
        names=FEATURE_COLUMNS_NAMES + [LABEL_COLUMN],
        dtype=merge_two_dicts(FEATURE_COLUMNS_DTYPE, LABEL_COLUMN_DTYPE),
    )
    os.remove(filename)

    # Define transformers for preprocessing
    numeric_features = list(FEATURE_COLUMNS_NAMES)
    numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())]
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

    # Apply transformations to data
    logger.info("Applying transforms")
    y = df.pop(LABEL_COLUMN)
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    dataset = np.concatenate((y_pre, X_pre), axis=1)

    logger.info(
        "Splitting %d rows of data into train, validation and test datasets", len(dataset))
    np.random.shuffle(dataset)
    train, validation, test = np.split(
        dataset, [int(0.7 * len(dataset)), int(0.85 * len(dataset))])

    # Write out preprocessed data to local file system
    output_rel_paths = {
        'train': 'train/train.csv',
        'validation': 'validation/validation.csv',
        'test': 'test/test.csv'
    }

    output_local_abs_paths = {
        k: f"{base_dir}/{v}" for k, v in output_rel_paths.items()}

    logger.info("Writing out datasets to %s", output_local_abs_paths['train'])
    pd.DataFrame(train).to_csv(
        output_local_abs_paths['train'], header=False, index=False)
    pd.DataFrame(validation).to_csv(
        output_local_abs_paths['validation'], header=False, index=False
    )
    pd.DataFrame
