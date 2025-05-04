from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich import box
from rich.panel import Panel

from review_clusterer.framework.csv_processor import CsvProcessor
from review_clusterer.framework.embedder import Embedder
from review_clusterer.framework.voyage_embedder import VoyageEmbedder
from review_clusterer.framework.local_embedder import LocalEmbedder
from review_clusterer.framework.chroma_repository import ChromaRepository


def index_controller(csv_file_path: Path, use_local_embedder: bool = False) -> None:
    """
    Controller for the index command. Processes a CSV file, creates embeddings,
    and saves to a ChromaDB vector database.

    Args:
        csv_file_path: Path to the CSV file
        use_local_embedder: If True, use the local sentence-transformers embedder
                           instead of VoyageAI. Default is False.
    """
    console = Console()

    try:
        console.print(
            Panel.fit(
                f"[bold]Indexing reviews from[/bold] [cyan]{csv_file_path.name}[/cyan]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

        # 1. Load and validate the CSV file
        console.print("[1/3] Loading and validating CSV file...", style="bold")
        processor = CsvProcessor(csv_file_path)

        # Check for missing required columns
        missing_columns = processor.validate()
        if missing_columns:
            raise ValueError(
                f"CSV file is missing required columns: {', '.join(missing_columns)}. "
                f"Required columns: {', '.join(processor.REQUIRED_COLUMNS)}"
            )

        # Clean the data
        data = processor.clean_data()
        review_count = len(data)
        console.print(f"  [green]✓[/green] Loaded and validated {review_count} reviews")

        # 2. Convert to list of dictionaries
        reviews = data.to_dict(orient="records")

        # 3. Create embeddings and set up database directory
        if use_local_embedder:
            console.print(
                "[2/3] Creating embeddings with local sentence-transformers model...",
                style="bold",
            )
            embedder = LocalEmbedder()
        else:
            console.print("[2/3] Creating embeddings with VoyageAI...", style="bold")
            embedder = VoyageEmbedder(
                api_key="pa-ZJzGbg--jB3Nq3dRz0cRPAhdLhCGzWeI1DNLxQfhnMP"
            )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding reviews", total=review_count)

            # Process reviews in batches to avoid API rate limits
            batch_size = 50
            embedded_reviews = []

            for i in range(0, review_count, batch_size):
                batch = reviews[i : i + batch_size]
                batch_embedded = embedder.create_review_embeddings(batch)
                embedded_reviews.extend(batch_embedded)
                progress.update(task, advance=len(batch))

        console.print(
            f"  [green]✓[/green] Created embeddings for {review_count} reviews"
        )

        # 4. Save to ChromaDB
        console.print("[3/3] Saving embeddings to ChromaDB...", style="bold")

        # Create new repository and add reviews
        collection_name, db_directory = ChromaRepository.get_paths_from_csv_file(
            csv_file_path, embedder.EMBEDDER_NAME
        )
        repository = ChromaRepository(
            collection_name, db_directory, delete_existing_collection=True
        )
        repository.add_reviews(embedded_reviews)

        console.print(
            f"  [green]✓[/green] Saved {review_count} reviews to ChromaDB at {db_directory}"
        )

        # Complete
        console.print(
            Panel.fit(
                f"[bold green]Successfully indexed[/bold green] [cyan]{review_count}[/cyan] reviews from [cyan]{csv_file_path.name}[/cyan].\n"
                f"Database saved to: [cyan]{db_directory}[/cyan]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise
