from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from review_clusterer.framework.csv_processor import CsvProcessor
from review_clusterer.framework.voyage_embedder import VoyageEmbedder
from review_clusterer.framework.local_embedder import LocalEmbedder
from review_clusterer.framework.chroma_repository import ChromaRepository


def format_search_result(result: Dict[str, Any], index: int) -> Panel:
    title = result.get("review_title", "No title")
    rating = result.get("review_rating", "No rating")
    details = result.get("review_details", "No details")
    reviewer = result.get("reviewer_name", "Anonymous")
    date = result.get("date", "No date")
    distance = result.get("distance", 0.0)

    metadata = Text.from_markup(
        f"[bold cyan]Rating:[/bold cyan] {rating}/5 stars | "
        f"[bold cyan]Reviewer:[/bold cyan] {reviewer} | "
        f"[bold cyan]Date:[/bold cyan] {date} | "
        f"[bold cyan]Similarity:[/bold cyan] {(1 - distance) * 100:.1f}%"
    )

    truncated_details = details[:200] + ("..." if len(details) > 200 else "")

    content = Text.from_markup(
        f"[bold]{title}[/bold]\n\n{truncated_details}\n\n{metadata}"
    )

    return Panel(
        content,
        title=f"[bold white]Result #{index + 1}[/bold white]",
        border_style="green",
        box=box.ROUNDED,
        title_align="left",
        padding=(1, 1),
    )


def search_controller(
    csv_file_path: Path, use_local_embedder: bool = False, top_n: int = 3
) -> None:
    console = Console()

    try:
        if use_local_embedder:
            embedder = LocalEmbedder()
        else:
            embedder = VoyageEmbedder()

        collection_name, db_directory = ChromaRepository.get_paths_from_csv_file(
            csv_file_path, embedder.EMBEDDER_NAME
        )

        if not db_directory.exists():
            console.print(
                f"[bold red]Error:[/bold red] Database not found at {db_directory}. "
                f"Please run 'index' command first."
            )
            return

        repository = ChromaRepository(collection_name, db_directory)

        console.print(
            Panel.fit(
                f"[bold]Interactive Search Mode[/bold] - Database: [cyan]{collection_name}[/cyan]",
                border_style="green",
                box=box.ROUNDED,
            )
        )
        console.print("Type your search query or [bold red]exit[/bold red] to quit.")

        while True:
            query = console.input("\n[bold cyan]Search query:[/bold cyan] ")

            if query.lower() in ("exit", "quit", "q"):
                console.print("[bold]Exiting search mode.[/bold]")
                break

            if not query.strip():
                continue

            console.print(f"[bold]Searching for:[/bold] {query}")

            query_embedding = embedder.create_embedding(query)

            results = repository.query_reviews(query_embedding, n_results=top_n)

            if not results or len(results["ids"][0]) == 0:
                console.print("[yellow]No results found.[/yellow]")
                continue

            console.print(
                f"\n[bold]Top {min(top_n, len(results['ids'][0]))} Results:[/bold]\n"
            )

            for i in range(len(results["ids"][0])):
                result = results["metadatas"][0][i].copy()
                result["distance"] = results["distances"][0][i]

                console.print(format_search_result(result, i))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise
