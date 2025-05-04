import os
from typing import List, Dict, Any, Optional, Union
import voyageai
from dotenv import load_dotenv
from pathlib import Path

from review_clusterer.framework.embedder import Embedder

class VoyageEmbedder(Embedder):
    """
    A wrapper for the VoyageAI API to create embeddings for review texts.
    """
    EMBEDDER_NAME = "voyage"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the VoyageAI embedder.
        
        Args:
            api_key: Optional VoyageAI API key. If not provided, will attempt to load from
                   environment variable VOYAGE_API_KEY or from a .env file.
        """
        load_dotenv()  # Load environment variables from .env file
        
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "VoyageAI API key is required. Provide it as an argument or "
                "set the VOYAGE_API_KEY environment variable."
            )
        
        self.client = voyageai.Client(api_key=self.api_key)
        self.model = "voyage-2"  # Using the latest Voyage model
    
    def format_review_text(self, title: str, rating: Union[int, float], content: str) -> str:
        """
        Format review text for embedding according to the specified format.
        
        Args:
            title: The review title
            rating: The review rating (usually out of 5)
            content: The review content/details
            
        Returns:
            Formatted review text string
        """
        return f"title:{title}\n{rating}/5 stars rating\ncontent:{content}"
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        response = self.client.embed(text, model=self.model).embeddings[0]
        return response
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: A list of texts to embed
            
        Returns:
            A list of embedding vectors
        """
        response = self.client.embed(texts, model=self.model).embeddings
        return response
    
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
        formatted_texts = [
            self.format_review_text(
                title=review.get("review_title", ""),
                rating=review.get("review_rating", 0),
                content=review.get("review_details", ""),
            )
            for review in reviews
        ]
        
        embeddings = self.create_embeddings(formatted_texts)
        
        # Add embeddings to the review dictionaries
        for i, (review, embedding, formatted_text) in enumerate(
            zip(reviews, embeddings, formatted_texts)
        ):
            reviews[i] = review.copy()  # Create a copy to avoid modifying the original
            reviews[i]["embedding"] = embedding
            reviews[i]["formatted_text"] = formatted_text
            
        return reviews