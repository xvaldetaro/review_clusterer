import click
from pathlib import Path
from review_clusterer.controllers.csv_controller import csv_test_controller
from review_clusterer.controllers.index_controller import index_controller
from review_clusterer.controllers.search_controller import search_controller

@click.group()
def cli():
    """Review Clusterer - A tool for analyzing customer reviews."""
    pass

@cli.command('csv-test')
@click.argument('csv_file_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
def csv_test(csv_file_path):
    """Test loading a CSV file by displaying the first 5 rows."""
    csv_test_controller(Path(csv_file_path))

@cli.command('index')
@click.argument('csv_file_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.option('--local', is_flag=True, help='Use local embedder instead of VoyageAI API')
def index(csv_file_path, local):
    """
    Process a CSV file, create embeddings, and save to a ChromaDB vector database.
    
    The database will be named after the CSV file (without extension) and will
    be stored in the same directory as the CSV file.
    
    By default, the embeddings are created using VoyageAI API. Use the --local
    flag to use a local sentence-transformers model instead.
    """
    index_controller(Path(csv_file_path), use_local_embedder=local)

@cli.command('search')
@click.argument('csv_file_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.option('--local', is_flag=True, help='Use local embedder instead of VoyageAI API')
@click.option('--top', default=3, help='Number of top results to display')
def search(csv_file_path, local, top):
    """
    Load a ChromaDB database matching the CSV file name and provide an interactive search interface.
    
    Enter search queries to find semantically similar reviews. The database must have been
    previously created with the 'index' command.
    
    Type 'exit' to exit the search mode.
    """
    search_controller(Path(csv_file_path), use_local_embedder=local, top_n=top)

if __name__ == "__main__":
    cli()