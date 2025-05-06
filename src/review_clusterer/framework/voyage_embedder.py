import os
from typing import List, Dict, Any, Optional, Union
import voyageai
from dotenv import load_dotenv
from pathlib import Path

from review_clusterer.framework.embedder import Embedder

class VoyageEmbedder(Embedder):
    EMBEDDER_NAME = "voyage"
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "VoyageAI API key is required. Provide it as an argument or "
                "set the VOYAGE_API_KEY environment variable."
            )
        
        self.client = voyageai.Client(api_key=self.api_key)
        self.model = "voyage-2"
    
    def format_review_text(self, title: str, rating: Union[int, float], content: str) -> str:
        return f"title:{title}\n{rating}/5 stars rating\ncontent:{content}"
    
    def create_embedding(self, text: str) -> List[float]:
        response = self.client.embed(text, model=self.model).embeddings[0]
        return response
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embed(texts, model=self.model).embeddings
        return response
    
    def create_review_embeddings(
        self, reviews: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        formatted_texts = [
            self.format_review_text(
                title=review.get("review_title", ""),
                rating=review.get("review_rating", 0),
                content=review.get("review_details", ""),
            )
            for review in reviews
        ]
        
        embeddings = self.create_embeddings(formatted_texts)
        
        for i, (review, embedding, formatted_text) in enumerate(
            zip(reviews, embeddings, formatted_texts)
        ):
            reviews[i] = review.copy()
            reviews[i]["embedding"] = embedding
            reviews[i]["formatted_text"] = formatted_text
            
        return reviews