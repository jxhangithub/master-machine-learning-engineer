import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import argparse
import os

def train_model(train_data_path, model_dir):
    # Read training data
    train_data = pd.read_csv(f"{train_data_path}/train.csv")

    # Split features and target
    X_train = train_data.drop('quality', axis=1)
    y_train = train_data['quality']

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    args = parser.parse_args()

    train_model(args.train_data, args.model_dir)