import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

import chromadb.errors

class ChromaRepository:
    """
    Repository for managing review embeddings in a ChromaDB collection.
    """

    def __init__(self, collection_name: str, persist_directory: Optional[Path] = None):
        """
        Initialize the ChromaDB repository.

        Args:
            collection_name: Name for the collection in the database
            persist_directory: Optional path to store the database. If None, will use an in-memory database.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize client with persistence settings
        if persist_directory:
            self.client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            self.client = chromadb.Client()

        # Create or get the collection
        self.recreate_collection()

    def recreate_collection(self):
        """
        Delete the collection if it exists and create a new one.
        """
        # Delete collection if it exists
        try:
            self.client.delete_collection(name=self.collection_name)
        except chromadb.errors.NotFoundError:
            # Collection doesn't exist yet, which is fine
            pass

        # Create a new collection
        self.collection = self.client.create_collection(name=self.collection_name)

    def delete_database(self) -> bool:
        """
        Delete the entire database directory if using persistent storage.

        Returns:
            True if database was deleted, False if using in-memory db
        """
        if self.persist_directory and self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
            return True
        return False

    def add_reviews(self, reviews: List[Dict[str, Any]]) -> None:
        """
        Add review embeddings to the ChromaDB collection.

        Args:
            reviews: A list of review dictionaries, each containing at minimum:
                   - 'id': Unique identifier for the review
                   - 'embedding': The embedding vector
                   - 'formatted_text': The formatted text that was embedded
                   - Additional metadata fields will be stored as is
        """
        if not reviews:
            return

        # Extract the required components for ChromaDB
        ids = [str(review["id"]) for review in reviews]
        embeddings = [review["embedding"] for review in reviews]
        documents = [review["formatted_text"] for review in reviews]

        # Extract metadata (exclude fields not needed in metadata)
        metadatas = []
        for review in reviews:
            metadata = {k: v for k, v in review.items()
                      if k not in ["embedding", "formatted_text"]}

            # Ensure all metadata values are strings or numbers
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)

            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query_reviews(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Query the collection for reviews similar to the query embedding.

        Args:
            query_embedding: Embedding vector to query with
            n_results: Number of results to return

        Returns:
            Query results containing ids, documents, and metadatas
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def get_all_reviews(self) -> Dict[str, Any]:
        """
        Get all reviews from the collection.

        Returns:
            All reviews in the collection with their embeddings
        """
        return self.collection.get(include=["embeddings", "documents", "metadatas"])

    def count(self) -> int:
        """
        Get the number of reviews in the collection.

        Returns:
            Number of reviews
        """
        return self.collection.count()