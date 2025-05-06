#!/usr/bin/env python
"""
Run with: poetry run python scripts/interactive_chroma.py <collection_name> <persist_directory>
"""

import sys
from pathlib import Path
from review_clusterer.framework.chroma_repository import ChromaRepository

def main():
    if len(sys.argv) < 3:
        print("Usage: poetry run python scripts/interactive_chroma.py <collection_name> <persist_directory>")
        sys.exit(1)
    csv_file_path = Path(sys.argv[1])
    embedder_name = sys.argv[2]
    base_collection_name = csv_file_path.stem
    collection_name = f"{base_collection_name}_{embedder_name}"
    persist_directory = csv_file_path.parent / collection_name

    if not persist_directory.exists():
        print(f"Error: Persist directory {persist_directory} does not exist.")
        sys.exit(1)

    repo = ChromaRepository(collection_name, persist_directory)
    print(f"Connected to ChromaDB collection: {collection_name}")
    print(f"Database location: {persist_directory}")
    print(f"Total reviews in collection: {repo.count()}")
    print("\nChromaRepository instance is available as 'repo'")
    print("Example commands:")
    print("  - repo.count()")
    print("  - results = repo.get_all_reviews()")
    print("  - repo.query_reviews([0.1, 0.2, ...], n_results=3)")

    import code
    code.interact(local=locals())

if __name__ == "__main__":
    main()