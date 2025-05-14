# Project Overview

This is a "Customer Review Analysis System" - a Python-based tool that processes customer reviews using embeddings to enable semantic search and clustering. This system helps businesses extract insights from reviews through vector-based analysis of text content.

## Functionality Breakdown:

1. Process and embed customer reviews from CSV files
   1.1. CSV files of reviews have the columns format `id,created_at,reviewer_name,date,review_title,review_details,review_rating,url`
2. Store embeddings in a vector database (chromaDB) for semantic search
3. Cluster reviews to identify common themes and sentiments using HDBSCAN
4. Search reviews using semantic similarity (distance in chromadb)
5. Visualize reviews with formatted output
6. (TODO) Cluster Refinement: Dedup, break apart, merge, summarize and annotate clusters using LLMs.
7. (TODO) Test alternative approach of Clustering reviews entirely by asking LLMs to do so. We start with all reviews being in the unclustered bucket and then run the same algorithm of (6. Cluster Refinement)
8. Parallel request multiple LLMs for each batch of reviews and get most reasonable result (with another LLM judge)

## Key Technologies

- **Embedding**: VoyageAI API (remote) or sentence-transformers (local)
- **Vector Database**: ChromaDB for storing and querying embeddings
- **Clustering**: HDBSCAN (or K-means) with optional UMAP reduction
- **LLM Integration**: OpenAI-compatible API for text analysis
- **CLI Interface**: Click library for command-line interface