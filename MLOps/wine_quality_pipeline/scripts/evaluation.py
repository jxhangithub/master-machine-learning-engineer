import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import argparse
import os

def evaluate_model(test_data_path, model_path, output_dir):
    # Load test data
    test_data = pd.read_csv(f"{test_data_path}/test.csv")

    # Split features and target
    X_test = test_data.drop('quality', axis=1)
    y_test = test_data['quality']

    # Load model
    model = joblib.load(os.path.join(model_path, "model.joblib"))

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Save metrics
    metrics = {
        'mse': float(mse),
        'r2': float(r2)
    }

    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    evaluate_model(args.test_data, args.model_path, args.output_dir)