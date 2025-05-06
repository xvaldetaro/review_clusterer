import pandas as pd
from pathlib import Path
from typing import Optional, List

class CsvProcessor:
    # Expected columns in the review CSV files
    EXPECTED_COLUMNS = [
        "id", "created_at", "reviewer_name", "date", 
        "review_title", "review_details", "review_rating", "url"
    ]
    
    # Required columns for creating embeddings
    REQUIRED_COLUMNS = ["id", "review_title", "review_details", "review_rating"]
    
    def __init__(self, csv_file_path: Path):
        self.csv_file_path = csv_file_path
        self._data: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
            
        self._data = pd.read_csv(self.csv_file_path)
        return self._data
    
    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            return self.load()
        return self._data
    
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        data = self.get_data()
        return data.head(n)
    
    def validate(self, strict: bool = False) -> List[str]:
        data = self.get_data()
        columns_to_check = self.EXPECTED_COLUMNS if strict else self.REQUIRED_COLUMNS
        missing_columns = [col for col in columns_to_check if col not in data.columns]
        return missing_columns
    
    def clean_data(self) -> pd.DataFrame:
        data = self.get_data()
        
        clean_data = data.dropna(subset=self.REQUIRED_COLUMNS)
        
        if "id" in clean_data.columns:
            clean_data["id"] = clean_data["id"].astype(str)
        
        if "review_rating" in clean_data.columns:
            clean_data["review_rating"] = pd.to_numeric(clean_data["review_rating"], errors="coerce")
            clean_data["review_rating"] = clean_data["review_rating"].fillna(0)
        
        return clean_data