from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

from review_clusterer.framework.csv_processor import CsvProcessor

def csv_test_controller(csv_file_path: Path) -> None:
    """
    Controller for the csv_test command. Loads and displays the first 5 rows of a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
    """
    console = Console()
    
    try:
        processor = CsvProcessor(csv_file_path)
        sample = processor.get_sample(5)
        
        # Create a rich table
        table = Table(title=f"Sample from {csv_file_path.name}", box=box.ROUNDED)
        
        # Add columns
        for column in sample.columns:
            table.add_column(column, style="cyan")
        
        # Add rows
        for _, row in sample.iterrows():
            table.add_row(*[str(val) for val in row.values])
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")