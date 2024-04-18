"""Module for data preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.model_selection import train_test_split


def main(input_folder: Path = typer.Option(...), output_folder: Path = typer.Option(...)) -> None:
    """Main function."""
    # Read data
    output_folder.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(input_folder / "student-por.csv")

    # Process grade
    df["average_grade"] = (df["G1"] + df["G2"] + df["G3"]) / 3
    df.drop(["G1", "G2", "G3"], inplace=True, axis=1)

    # Encode categorical
    binary_cols = [col for col in df.columns if len(df[col].unique()) == 2]
    n_unique_by_column = df.nunique()
    binary_cols = n_unique_by_column[n_unique_by_column == 2].index
    df = pd.get_dummies(df, columns=binary_cols, drop_first=True, dtype=float)  # maybe drop_first False or other method
    discrete_features = [col for col, dtype in df.dtypes.items() if not np.issubdtype(dtype, np.number)]
    df = pd.get_dummies(df, columns=discrete_features, dtype=float, drop_first=False)

    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train, df_val = train_test_split(df_train, test_size=0.2)
    print(f"{df_train.shape=}, {df_val.shape=}, {df_test.shape=}")

    # Save data
    df_train.to_parquet(output_folder / "train_data.parquet", index=False)
    df_val.to_parquet(output_folder / "val_data.parquet", index=False)
    df_test.to_parquet(output_folder / "test_data.parquet", index=False)


if __name__ == "__main__":
    typer.run(main)
