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
from review_clusterer.controllers.llm_controller import (
    llm_test_controller,
    llm_structured_test_controller,
)


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
    """Process a CSV file, create embeddings, and save to a ChromaDB vector database."""
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
    """Interactive search interface for finding semantically similar reviews."""
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
@click.option(
    "--output-markdown", is_flag=True,
    help="Generate a markdown report instead of console output"
)
@click.option(
    "--output-path", type=click.Path(file_okay=True, dir_okay=False, writable=True),
    help="Path where to save the markdown report (only used with --output-markdown)"
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
    output_markdown,
    output_path,
):
    """Cluster reviews based on their embeddings and display the results."""
    if hdbscan and clusters is not None:
        click.echo("Warning: When using HDBSCAN, the 'clusters' argument is ignored as cluster count is determined automatically")

    output_path_obj = Path(output_path) if output_path else None

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
        output_markdown=output_markdown,
        output_path=output_path_obj,
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
    """Plot the elbow method to determine the optimal number of clusters."""
    plot_cluster_distribution(Path(csv_file_path), use_local_embedder=local)


@cli.command("llm-test")
@click.option("--provider", type=click.Choice(["openai", "anthropic"]), default="openai", help="LLM provider to use")
@click.option("--model", help="Model name (defaults to gpt-4o for OpenAI or claude-3-opus for Anthropic)")
@click.option("--prompt", help="Prompt to send to the LLM", default="Summarize the key points of effective customer service.")
@click.option(
    "--api-key-file", 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to a file containing the API key"
)
def llm_test(provider, model, prompt, api_key_file):
    """Test the LLM client with a simple prompt."""
    api_key_path = Path(api_key_file) if api_key_file else None
    llm_test_controller(provider, prompt, model, api_key_path)


@cli.command("llm-structured-test")
@click.option("--provider", type=click.Choice(["openai", "anthropic"]), default="openai", help="LLM provider to use")
@click.option("--model", help="Model name (defaults to gpt-4o for OpenAI or claude-3-opus for Anthropic)")
@click.option(
    "--prompt", 
    help="Prompt to send to the LLM", 
    default="Analyze these customer reviews for sentiment and key themes: 'The product exceeded my expectations, it's fast and reliable.', 'Terrible customer service, had to wait for hours.'"
)
@click.option(
    "--schema-file", 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to a JSON file containing the response schema"
)
@click.option(
    "--api-key-file", 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to a file containing the API key"
)
def llm_structured_test(provider, model, prompt, schema_file, api_key_file):
    """Test the LLM client with a structured output request."""
    import json
    
    schema = None
    if schema_file:
        with open(schema_file, 'r') as f:
            schema = json.load(f)
    
    api_key_path = Path(api_key_file) if api_key_file else None
    llm_structured_test_controller(provider, prompt, schema, model, api_key_path)


if __name__ == "__main__":
    cli()
