"""Module for training the model."""

from pathlib import Path

import typer


def main(preprocess_folder: Path = typer.Option(...), output_folder: Path = typer.Option(...)) -> None:
    """Main function."""


if __name__ == "__main__":
    typer.run(main)
