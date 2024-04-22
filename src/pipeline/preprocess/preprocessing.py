"""Module for data preprocessing."""

from pathlib import Path

import typer


def main(input_folder: Path = typer.Option(...), output_folder: Path = typer.Option(...)) -> None:
    """Main function."""


if __name__ == "__main__":
    typer.run(main)
