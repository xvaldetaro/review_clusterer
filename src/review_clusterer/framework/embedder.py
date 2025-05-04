from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class Embedder(ABC):
    """
    Abstract base class that defines the interface for embedding review texts.
    """
    EMBEDDER_NAME: str = "base"  # Should be overridden by child classes
    
    @abstractmethod
    def format_review_text(self, title: str, rating: Union[int, float], content: str) -> str:
        """
        Format review text for embedding according to a specific format.
        
        Args:
            title: The review title
            rating: The review rating (usually out of 5)
            content: The review content/details
            
        Returns:
            Formatted review text string
        """
        pass
    
    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        pass
    
    @abstractmethod
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: A list of texts to embed
            
        Returns:
            A list of embedding vectors
        """
        pass
    
    @abstractmethod
    def create_review_embeddings(
        self, reviews: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create embeddings for a list of review dictionaries.
        
        Args:
            reviews: A list of review dictionaries, each containing 'review_title',
                   'review_rating', and 'review_details' keys.
            
        Returns:
            A list of dictionaries containing the original reviews with added 'embedding' and
            'formatted_text' keys.
        """
        pass