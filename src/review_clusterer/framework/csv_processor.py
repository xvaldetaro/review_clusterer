import pandas as pd
from pathlib import Path
from typing import Optional, List

class CsvProcessor:
    """
    Processor for handling CSV files containing review data.
    """
    
    # Expected columns in the review CSV files
    EXPECTED_COLUMNS = [
        "id", "created_at", "reviewer_name", "date", 
        "review_title", "review_details", "review_rating", "url"
    ]
    
    # Required columns for creating embeddings
    REQUIRED_COLUMNS = ["id", "review_title", "review_details", "review_rating"]
    
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
    
    def validate(self, strict: bool = False) -> List[str]:
        """
        Validate that the CSV has the expected columns.
        
        Args:
            strict: If True, checks for all expected columns. If False, only checks for required columns.
            
        Returns:
            List of missing columns. Empty list if all required/expected columns are present.
        """
        data = self.get_data()
        columns_to_check = self.EXPECTED_COLUMNS if strict else self.REQUIRED_COLUMNS
        missing_columns = [col for col in columns_to_check if col not in data.columns]
        return missing_columns
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the data by removing rows with missing values in required columns
        and ensuring proper data types.
        
        Returns:
            Cleaned DataFrame
        """
        data = self.get_data()
        
        # Drop rows with missing values in required columns
        clean_data = data.dropna(subset=self.REQUIRED_COLUMNS)
        
        # Ensure id is a string
        if "id" in clean_data.columns:
            clean_data["id"] = clean_data["id"].astype(str)
        
        # Ensure review_rating is a float
        if "review_rating" in clean_data.columns:
            clean_data["review_rating"] = pd.to_numeric(clean_data["review_rating"], errors="coerce")
            # Fill NaN values with 0
            clean_data["review_rating"] = clean_data["review_rating"].fillna(0)
        
        return clean_data