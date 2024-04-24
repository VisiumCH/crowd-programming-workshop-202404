"""Module for training the model."""

import pickle
from pathlib import Path

import pandas as pd
import typer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def main(preprocess_folder: Path = typer.Option(...), output_folder: Path = typer.Option(...)) -> None:
    """Main function."""
    output_folder.mkdir(exist_ok=True, parents=True)

    df_train = pd.read_parquet(preprocess_folder / "df_train.parquet")
    df_val = pd.read_parquet(preprocess_folder / "df_val.parquet")

    model = RandomForestRegressor()

    target_column = "average_grade"
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    model.fit(X_train, y_train)

    X_val = df_val.drop(columns=[target_column])
    y_val = df_val[target_column]
    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    print(mae)
    with open(output_folder / "model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    typer.run(main)
