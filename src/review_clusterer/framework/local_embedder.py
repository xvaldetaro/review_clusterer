from typing import List, Dict, Any, Union, Optional
from sentence_transformers import SentenceTransformer

from review_clusterer.framework.embedder import Embedder


class LocalEmbedder(Embedder):
    """
    A local embedder using sentence-transformers with the MiniLM model.
    """
    EMBEDDER_NAME = "minilm"
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local embedder with a sentence-transformer model.
        
        Args:
            model_name: The name of the pretrained model to use.
                        Default is "all-MiniLM-L6-v2" which is a small, fast model
                        with good performance for semantic similarity tasks.
        """
        self.model = SentenceTransformer(model_name)
    
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
        embedding = self.model.encode(text)
        return embedding.tolist()  # Convert numpy array to list
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: A list of texts to embed
            
        Returns:
            A list of embedding vectors
        """
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]  # Convert numpy arrays to lists
    
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