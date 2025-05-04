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
    """
    Format a single search result into a rich Panel with a compact layout.
    
    Args:
        result: A search result dictionary containing metadata
        index: The position in the result list
        
    Returns:
        A rich Panel object displaying the result in a compact format
    """
    # Extract metadata for display
    title = result.get("review_title", "No title")
    rating = result.get("review_rating", "No rating")
    details = result.get("review_details", "No details")
    reviewer = result.get("reviewer_name", "Anonymous")
    date = result.get("date", "No date")
    distance = result.get("distance", 0.0)
    
    # Format metadata in a compact horizontal layout
    metadata = Text.from_markup(
        f"[bold cyan]Rating:[/bold cyan] {rating}/5 stars | "
        f"[bold cyan]Reviewer:[/bold cyan] {reviewer} | "
        f"[bold cyan]Date:[/bold cyan] {date} | "
        f"[bold cyan]Similarity:[/bold cyan] {(1 - distance) * 100:.1f}%"
    )
    
    # Create a compact panel with the title, metadata, and truncated details
    # Limit details to 200 characters and add ellipsis if longer
    truncated_details = details[:200] + ("..." if len(details) > 200 else "")
    
    content = Text.from_markup(
        f"[bold]{title}[/bold]\n\n"
        f"{truncated_details}\n\n"
        f"{metadata}"
    )
    
    return Panel(
        content,
        title=f"[bold white]Result #{index + 1}[/bold white]",
        border_style="green",
        box=box.ROUNDED,
        title_align="left",
        padding=(1, 1)  # Reduced padding for more compact display
    )


def search_controller(
    csv_file_path: Path, 
    use_local_embedder: bool = False,
    top_n: int = 3
) -> None:
    """
    Controller for the search command. Loads a ChromaDB database matching the CSV file name
    and provides an interactive search interface.
    
    Args:
        csv_file_path: Path to the CSV file that was indexed
        use_local_embedder: If True, use the local sentence-transformers embedder
                          instead of VoyageAI. Default is False.
        top_n: Number of top results to display
    """
    console = Console()
    
    try:
        # Create base collection name from CSV filename (without extension)
        base_collection_name = csv_file_path.stem
        
        # Initialize embedder (same as in index_controller)
        if use_local_embedder:
            embedder = LocalEmbedder()
        else:
            embedder = VoyageEmbedder()
        
        # Create collection name with embedding type (must match what was used in indexing)
        collection_name = f"{base_collection_name}_{embedder.EMBEDDER_NAME}"
        db_directory = csv_file_path.parent / collection_name
        
        # Check if the database exists
        if not db_directory.exists():
            console.print(
                f"[bold red]Error:[/bold red] Database not found at {db_directory}. "
                f"Please run 'index' command first."
            )
            return
        
        # Connect to the ChromaDB repository
        repository = ChromaRepository(collection_name, db_directory)
        
        console.print(Panel.fit(
            f"[bold]Interactive Search Mode[/bold] - Database: [cyan]{collection_name}[/cyan]",
            border_style="green",
            box=box.ROUNDED
        ))
        console.print("Type your search query or [bold red]exit[/bold red] to quit.")
        
        # Interactive search loop
        while True:
            # Get user query
            query = console.input("\n[bold cyan]Search query:[/bold cyan] ")
            
            # Exit condition
            if query.lower() in ("exit", "quit", "q"):
                console.print("[bold]Exiting search mode.[/bold]")
                break
                
            # Skip empty queries
            if not query.strip():
                continue
                
            console.print(f"[bold]Searching for:[/bold] {query}")
            
            # Create embedding for the query
            query_embedding = embedder.create_embedding(query)
            
            # Query the database
            results = repository.query_reviews(query_embedding, n_results=top_n)
            
            if not results or len(results["ids"][0]) == 0:
                console.print("[yellow]No results found.[/yellow]")
                continue
                
            # Display results
            console.print(f"\n[bold]Top {min(top_n, len(results['ids'][0]))} Results:[/bold]\n")
            
            for i in range(len(results["ids"][0])):
                # Combine metadata with distance for display
                result = results["metadatas"][0][i].copy()
                result["distance"] = results["distances"][0][i]
                
                # Display the formatted result
                console.print(format_search_result(result, i))
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise