"""Module for data preprocessing."""

from pathlib import Path

import typer


def main(input_folder: Path = typer.Option(...), output_folder: Path = typer.Option(...)) -> None:
    """Main function."""
    output_folder.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    typer.run(main)
