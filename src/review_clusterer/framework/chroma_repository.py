import chromadb
from rich.console import Console
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional
from chromadb.utils.batch_utils import create_batches
import shutil

import chromadb.errors

console = Console()


class ChromaRepository:
    def get_paths_from_csv_file(
        csv_file_path: Path, embedder_name: str
    ) -> tuple[str, Path]:
        base_collection_name = csv_file_path.stem
        collection_name = f"{base_collection_name}_{embedder_name}"
        db_directory = csv_file_path.parent / collection_name
        return (collection_name, db_directory)

    def __init__(
        self,
        collection_name: str,
        persist_directory: Path,
        delete_existing_collection: bool = False,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.client = chromadb.PersistentClient(
            path=str(persist_directory), settings=Settings(allow_reset=True)
        )

        if delete_existing_collection:
            console.print("Deleting existing collection...")
            try:
                self.client.delete_collection(name=self.collection_name)
            except chromadb.errors.NotFoundError:
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def delete_database(directory: Path) -> bool:
        shutil.rmtree(directory)

    def add_reviews(self, reviews: List[Dict[str, Any]]) -> None:
        if not reviews:
            return

        ids = [str(review["id"]) for review in reviews]
        embeddings = [review["embedding"] for review in reviews]
        documents = [review["formatted_text"] for review in reviews]

        metadatas = []
        for review in reviews:
            metadata = {
                k: v
                for k, v in review.items()
                if k not in ["embedding", "formatted_text"]
            }

            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)

            metadatas.append(metadata)

        batches = create_batches(
            api=self.client,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        for batch in batches:
            self.collection.add(
                ids=batch[0],
                documents=batch[3],
                embeddings=batch[1],
                metadatas=batch[2],
            )

    def query_reviews(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Dict[str, Any]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        return results

    def get_all_reviews(self) -> Dict[str, Any]:
        return self.collection.get(include=["embeddings", "documents", "metadatas"])

    def count(self) -> int:
        return self.collection.count()
