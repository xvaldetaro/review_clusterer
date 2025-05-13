# Project Overview

This is a "Customer Review Analysis System" - a Python-based tool that processes customer reviews using embeddings to enable semantic search and clustering. This system helps businesses extract insights
from reviews through vector-based analysis of text content.

Functionality Breakdown:

1. Process and embed customer reviews from CSV files
   1.1. CSV files of reviews have the columns format `id,created_at,reviewer_name,date,review_title,review_details,review_rating,url`
2. Store embeddings in a vector database (chromaDB) for semantic search
3. Cluster reviews to identify common themes and sentiments using HDBSCAN
4. Search reviews using semantic similarity (distance in chromadb)
5. Visualize reviews with formatted output
6. (TODO) Analyze clusters using LLMs to generate insights and summaries
7. (TODO) Test alternative approach of Clustering reviews entirely by asking LLMs to do so. Probably send small batches of reviews and a summary of all existing clusters (starts with 0). LLM responds in JSON telling which cluster each review goes to.
7.1. Parallel request multiple LLMs for each batch of reviews and get most reasonable result (with another LLM judge)

# Project Details
