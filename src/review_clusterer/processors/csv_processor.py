import pandas as pd
from pathlib import Path
from typing import Optional

class CsvProcessor:
    """
    Processor for handling CSV files containing review data.
    """
    
    def __init__(self, csv_file_path: Path):
        """
        Initialize the CSV processor with a file path.
        
        Args:
            csv_file_path: Path to the CSV file
        """
        self.csv_file_path = csv_file_path
        self._data: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        """
        Load the CSV file into a pandas DataFrame.
        
        Returns:
            DataFrame containing the CSV data
        """
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
            
        self._data = pd.read_csv(self.csv_file_path)
        return self._data
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded data or load it if not already loaded.
        
        Returns:
            DataFrame containing the CSV data
        """
        if self._data is None:
            return self.load()
        return self._data
    
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Get a sample of n rows from the CSV data.
        
        Args:
            n: Number of rows to return
            
        Returns:
            DataFrame containing n rows from the CSV data
        """
        data = self.get_data()
        return data.head(n)