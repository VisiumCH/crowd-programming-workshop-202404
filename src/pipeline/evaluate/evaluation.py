"""Module for model evaluation."""

import pickle
from pathlib import Path

import pandas as pd
import typer
from sklearn.metrics import root_mean_squared_error

from src.pipeline.train.training import split_feature_and_label


def main(preprocess_folder: Path = typer.Option(...), train_folder: Path = typer.Option(...)) -> None:
    """Main function."""
    df_test = pd.read_parquet(preprocess_folder / "test_data.parquet")
    df_train = pd.read_parquet(preprocess_folder / "train_data.parquet")

    X_test, y_test = split_feature_and_label(df_test)
    _, y_train = split_feature_and_label(df_train)

    with open(train_folder / "model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_baseline = [y_train.mean()] * len(y_test)

    test_metric = root_mean_squared_error(y_test, y_pred)
    baseline_metric = root_mean_squared_error(y_test, y_baseline)

    print(f"Test RMSE: {test_metric:.2f}")
    print(f"Baseline RMSE: {baseline_metric:.2f}")


if __name__ == "__main__":
    typer.run(main)
