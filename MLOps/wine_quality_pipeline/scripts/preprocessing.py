import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os

def preprocess_data(input_data_path, output_data_path):
    # Read the data
    df = pd.read_csv(input_data_path)

    # Basic cleaning
    df = df.dropna()

    # Feature engineering
    df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
    df['acid_ratio'] = df['fixed acidity'] / df['volatile acidity']

    # Split features and target
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Save processed datasets
    train_data = pd.concat([pd.DataFrame(X_train), pd.Series(y_train)], axis=1)
    test_data = pd.concat([pd.DataFrame(X_test), pd.Series(y_test)], axis=1)

    train_data.to_csv(f"{output_data_path}/train.csv", index=False)
    test_data.to_csv(f"{output_data_path}/test.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-data", type=str, required=True)
    args = parser.parse_args()

    preprocess_data(args.input_data, args.output_data)