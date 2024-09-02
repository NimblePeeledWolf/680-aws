from __future__ import print_function

import argparse
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA', '/opt/ml/output/data'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    
    args = parser.parse_args()
    
    file = os.path.join(args.train, "wine_quality_data.csv")
    dataset = pd.read_csv(file, engine="python")
    
    x = dataset.drop("target", axis=1)
    y = dataset["target"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    
    joblib.dump(regressor, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    # Deserialize and return the fitted model; name should match the serialized model in the main method
    regressor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return regressor
