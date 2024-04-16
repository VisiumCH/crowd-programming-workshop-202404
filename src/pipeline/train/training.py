"""Module for training the model."""

import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import typer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def split_feature_and_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split feature and label columns.

    Args:
        df (pd.DataFrame): Dataframe to split

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Tuple with features dataframe and label column
    """
    X = df.drop(columns="average_grade")
    y = df["average_grade"]
    return X, y


def main(preprocess_folder: Path = typer.Option(...), output_folder: Path = typer.Option(...)) -> None:
    """Main function."""
    # Load Data
    df_train = pd.read_csv(preprocess_folder / "train_data.csv")
    df_val = pd.read_csv(preprocess_folder / "val_data.csv")

    X_train, y_train = split_feature_and_label(df_train)
    X_val, y_val = split_feature_and_label(df_val)

    # Create Model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Compute evaluation metric
    metric = root_mean_squared_error(y_val, y_pred)
    print(f"Validation RMSE: {metric:.2f}")

    # Save outputs
    output_folder.mkdir(exist_ok=True, parents=True)
    with open(output_folder / "model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    typer.run(main)
