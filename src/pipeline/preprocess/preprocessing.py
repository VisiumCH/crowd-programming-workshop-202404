"""Module for data preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.model_selection import train_test_split


def main(input_folder: Path = typer.Option(...), output_folder: Path = typer.Option(...)) -> None:
    """Main function."""
    output_folder.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(input_folder / "student-por.csv")

    categorical_variables = [col for col, dtype in df.items() if not np.issubdtype(dtype, np.number)]
    df = pd.get_dummies(df, columns=categorical_variables, drop_first=False, dtype=float)

    df["average_grade"] = (df["G1"] + df["G2"] + df["G3"]) / 3.0
    df = df.drop(["G1", "G2", "G3"], axis="columns")

    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train, df_val = train_test_split(df_train, test_size=0.2)

    df_train.to_parquet(output_folder / "df_train.parquet")
    df_val.to_parquet(output_folder / "df_val.parquet")
    df_test.to_parquet(output_folder / "df_test.parquet")


if __name__ == "__main__":
    typer.run(main)
