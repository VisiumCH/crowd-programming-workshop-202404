"""Module for model evaluation."""

import pickle
from pathlib import Path
from statistics import mean

import pandas as pd
import typer
from sklearn.metrics import mean_absolute_error


def main(preprocess_folder: Path = typer.Option(...), train_folder: Path = typer.Option(...)) -> None:
    """Main function."""
    df_train = pd.read_parquet(preprocess_folder / "df_train.parquet")
    df_test = pd.read_parquet(preprocess_folder / "df_test.parquet")

    with open(train_folder / "model.pkl", "rb") as f:
        model = pickle.load(f)
    target_column = "average_grade"
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    model.fit(X_train, y_train)

    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    predictions = model.predict(X_test)

    baseline_pred = y_train.mean()

    mae = mean_absolute_error(y_test, predictions)
    mae_baseline = mean_absolute_error(y_test, [baseline_pred] * len(y_test))

    print(f"{mae=}, {mae_baseline=}")


if __name__ == "__main__":
    typer.run(main)
