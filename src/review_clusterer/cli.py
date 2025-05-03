import click
from pathlib import Path
from review_clusterer.controllers.csv_controller import csv_test_controller

@click.group()
def cli():
    """Review Clusterer - A tool for analyzing customer reviews."""
    pass

@cli.command('csv-test')
@click.argument('csv_file_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
def csv_test(csv_file_path):
    """Test loading a CSV file by displaying the first 5 rows."""
    csv_test_controller(Path(csv_file_path))

if __name__ == "__main__":
    cli()