import click
from pathlib import Path
from review_clusterer.controllers.csv_controller import csv_test_controller
from review_clusterer.controllers.index_controller import index_controller
from review_clusterer.controllers.search_controller import search_controller
from review_clusterer.controllers.cluster_controller import (
    cluster_controller,
    plot_cluster_distribution,
)
from review_clusterer.controllers.cluster_controller import plot_elbow


@click.group()
def cli():
    """Review Clusterer - A tool for analyzing customer reviews."""
    pass


@cli.command("csv-test")
@click.argument(
    "csv_file_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
def csv_test(csv_file_path):
    """Test loading a CSV file by displaying the first 5 rows."""
    csv_test_controller(Path(csv_file_path))


@cli.command("index")
@click.argument(
    "csv_file_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--local", is_flag=True, help="Use local embedder instead of VoyageAI API"
)
def index(csv_file_path, local):
    """
    Process a CSV file, create embeddings, and save to a ChromaDB vector database.

    The database will be named after the CSV file (without extension) and will
    be stored in the same directory as the CSV file.

    By default, the embeddings are created using VoyageAI API. Use the --local
    flag to use a local sentence-transformers model instead.
    """
    index_controller(Path(csv_file_path), use_local_embedder=local)


@cli.command("search")
@click.argument(
    "csv_file_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--local", is_flag=True, help="Use local embedder instead of VoyageAI API"
)
@click.option("--top", default=3, help="Number of top results to display")
def search(csv_file_path, local, top):
    """
    Load a ChromaDB database matching the CSV file name and provide an interactive search interface.

    Enter search queries to find semantically similar reviews. The database must have been
    previously created with the 'index' command.

    Type 'exit' to exit the search mode.
    """
    search_controller(Path(csv_file_path), use_local_embedder=local, top_n=top)


@cli.command("cluster")
@click.argument(
    "csv_file_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.argument("clusters", type=int, required=False)
@click.option(
    "--local", is_flag=True, help="Use local embedder instead of VoyageAI API"
)
@click.option(
    "--hdbscan", is_flag=True, 
    help="Use HDBSCAN clustering instead of K-means. This algorithm automatically detects outliers."
)
@click.option(
    "--min-cluster-size", type=int, default=10,
    help="Minimum number of points to form a cluster (HDBSCAN only)"
)
@click.option(
    "--min-samples", type=int, default=5,
    help="Controls the resilience to noise (HDBSCAN only)"
)
@click.option(
    "--no-umap", is_flag=True,
    help="Disable UMAP dimensionality reduction (HDBSCAN only)"
)
@click.option(
    "--umap-neighbors", type=int, default=15,
    help="Number of neighbors for UMAP (HDBSCAN only)"
)
@click.option(
    "--umap-components", type=int, default=10,
    help="Number of components for dimensionality reduction (HDBSCAN only)"
)
def cluster(
    csv_file_path, 
    clusters, 
    local, 
    hdbscan, 
    min_cluster_size, 
    min_samples,
    no_umap,
    umap_neighbors,
    umap_components,
):
    """
    Cluster reviews based on their embeddings and display the results.

    This command will:
    1. Load a ChromaDB database matching the CSV file name
    2. Run a clustering algorithm to group similar reviews
    3. Display the resulting clusters sorted by average review rating (worst to best)
       with statistics and representative reviews

    The database must have been previously created with the 'index' command.

    Two clustering methods are available:
    - K-means (default): Requires number of clusters, assigns all reviews to clusters
    - HDBSCAN (with --hdbscan): Automatically finds clusters, identifies outliers

    When using HDBSCAN, you can adjust:
    - min-cluster-size: Minimum reviews to form a cluster (default: 10)
    - min-samples: Controls how conservative clustering is (default: 5)
    - UMAP dimensionality reduction (enabled by default)
    """
    if hdbscan and clusters is not None:
        click.echo("Warning: When using HDBSCAN, the 'clusters' argument is ignored as cluster count is determined automatically")

    cluster_controller(
        Path(csv_file_path), 
        n_clusters=clusters, 
        use_local_embedder=local,
        use_hdbscan=hdbscan,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        use_umap=not no_umap,
        umap_n_neighbors=umap_neighbors,
        umap_n_components=umap_components,
    )


@cli.command("plot-elbow")
@click.argument(
    "csv_file_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--local", is_flag=True, help="Use local embedder instead of VoyageAI API"
)
def do_plot_elbow(csv_file_path, local):
    """
    Cluster reviews based on their embeddings and display the results.

    This command will:
    1. Load a ChromaDB database matching the CSV file name
    2. Determine the optimal number of clusters (if not specified)
    3. Run a clustering algorithm to group similar reviews
    4. Display the resulting clusters sorted by average review rating (worst to best)
       with statistics and representative reviews

    The database must have been previously created with the 'index' command.
    """
    plot_cluster_distribution(Path(csv_file_path), use_local_embedder=local)


if __name__ == "__main__":
    cli()
