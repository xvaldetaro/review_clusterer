from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

from review_clusterer.framework.csv_processor import CsvProcessor

def csv_test_controller(csv_file_path: Path) -> None:
    console = Console()
    
    try:
        processor = CsvProcessor(csv_file_path)
        sample = processor.get_sample(5)
        
        table = Table(title=f"Sample from {csv_file_path.name}", box=box.ROUNDED)
        
        for column in sample.columns:
            table.add_column(column, style="cyan")
        
        for _, row in sample.iterrows():
            table.add_row(*[str(val) for val in row.values])
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")