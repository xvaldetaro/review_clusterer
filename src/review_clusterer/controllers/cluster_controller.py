from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import time

from review_clusterer.framework.chroma_repository import ChromaRepository
from review_clusterer.framework.clusterer import cluster_reviews
from review_clusterer.framework.clusterer import plot_elbow
from review_clusterer.framework.voyage_embedder import VoyageEmbedder
from review_clusterer.framework.local_embedder import LocalEmbedder

console = Console()


def get_embeddings(csv_file_path: Path, use_local_embedder: bool = False):
    embedder_name = (
        LocalEmbedder.EMBEDDER_NAME
        if use_local_embedder
        else VoyageEmbedder.EMBEDDER_NAME
    )
    collection_name, db_directory = ChromaRepository.get_paths_from_csv_file(
        csv_file_path, embedder_name
    )

    # Check if the database exists
    if not db_directory.exists():
        console.print(
            f"[red]Error: ChromaDB database not found at {db_directory}[/red]"
        )
        console.print(
            f"[yellow]Please run 'review-clusterer index {csv_file_path}' first to create the database.[/yellow]"
        )
        return

    # Initialize repository
    console.print(f"[green]Loading ChromaDB collection: {collection_name}[/green]")
    repo = ChromaRepository(
        collection_name=collection_name, persist_directory=db_directory
    )

    # Check if the collection has any reviews
    review_count = repo.count()
    if review_count == 0:
        console.print(f"[red]Error: Collection '{collection_name}' is empty[/red]")
        return

    console.print(f"[green]Found {review_count} reviews in the collection[/green]")

    # Get all reviews with embeddings
    console.print("[green]Loading review embeddings...[/green]")
    all_reviews = repo.get_all_reviews()

    # Convert to list of review dictionaries
    reviews_with_embeddings = []
    for i, (review_id, embedding, document, metadata) in enumerate(
        zip(
            all_reviews["ids"],
            all_reviews["embeddings"],
            all_reviews["documents"],
            all_reviews["metadatas"],
        )
    ):
        review_dict = {
            "id": review_id,
            "embedding": embedding,
            "formatted_text": document,
            **metadata,
        }
        reviews_with_embeddings.append(review_dict)

    return reviews_with_embeddings


def plot_cluster_distribution(
    csv_file_path: Path, use_local_embedder: bool = False
) -> None:
    reviews_with_embeddings = get_embeddings(csv_file_path, use_local_embedder)
    plot_elbow(reviews_with_embeddings)


def cluster_controller(
    csv_file_path: Path, n_clusters: int, use_local_embedder: bool = False
) -> None:
    start_time = time.time()

    reviews_with_embeddings = get_embeddings(csv_file_path, use_local_embedder)

    console.print(
        f"[green]Clustering {len(reviews_with_embeddings)} reviews...[/green]"
    )
    clusters = cluster_reviews(reviews_with_embeddings, n_clusters)
    
    console.print(
        f"[green]Displaying clusters sorted by average rating (worst to best)...[/green]"
    )
    
    # Display clusters
    display_clusters(clusters)

    elapsed_time = time.time() - start_time
    console.print(f"[green]Clustering completed in {elapsed_time:.2f} seconds[/green]")


def display_clusters(clusters: list) -> None:
    """
    Display the clusters in a formatted output, sorted by average rating (worst to best).

    Args:
        clusters: List of cluster dictionaries sorted by average rating
    """
    for i, cluster in enumerate(clusters):
        # Create a table for this cluster
        table = Table(
            title=f"Cluster {i + 1}/{len(clusters)} (ID: {cluster['id']}): {cluster['review_count']} reviews, "
            f"Mean distance: {cluster['mean_distance']:.4f}, "
            f"Avg rating: {cluster['avg_rating']:.1f}/5"
        )

        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Rating", style="magenta")
        table.add_column("Distance", style="green")
        table.add_column("Title", style="blue")
        table.add_column("Content", style="white")

        # Get the 5 most central reviews
        central_reviews = cluster["reviews"][:5]

        # Add rows for central reviews
        for review in central_reviews:
            review_id = review["id"]
            try:
                rating = f"{float(review.get('review_rating', 0)):.1f}/5"
            except (ValueError, TypeError):
                rating = "N/A"

            distance = f"{review.get('distance_from_center', 0):.4f}"
            title = review.get("review_title", "")
            content = review.get("review_details", "")

            # Truncate content if too long
            if len(content) > 100:
                content = content[:97] + "..."

            table.add_row(str(review_id), rating, distance, title, content)

        # Display the table
        console.print(table)
        console.print("\n")
