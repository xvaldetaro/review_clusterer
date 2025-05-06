from typing import List, Dict, Any, Union, Optional
from sentence_transformers import SentenceTransformer

from review_clusterer.framework.embedder import Embedder


class LocalEmbedder(Embedder):
    EMBEDDER_NAME = "minilm"
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def format_review_text(self, title: str, rating: Union[int, float], content: str) -> str:
        return f"title:{title}\n{rating}/5 stars rating\ncontent:{content}"
    
    def create_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]
    
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