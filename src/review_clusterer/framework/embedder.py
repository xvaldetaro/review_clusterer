from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class Embedder(ABC):
    EMBEDDER_NAME: str = "base"
    
    @abstractmethod
    def format_review_text(self, title: str, rating: Union[int, float], content: str) -> str:
        pass
    
    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def create_review_embeddings(
        self, reviews: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        pass