"""Module for model evaluation."""

from pathlib import Path

import typer


def main(preprocess_folder: Path = typer.Option(...), train_folder: Path = typer.Option(...)) -> None:
    """Main function."""


if __name__ == "__main__":
    typer.run(main)
