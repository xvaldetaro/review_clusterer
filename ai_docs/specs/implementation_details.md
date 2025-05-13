# Project Implementation Details

This file contains detailed information about the project architecture, components and data flow. Include this file in your prompts when:

1. You need Claude to understand implementation specifics before making structural changes
2. You're adding new features that need to integrate with existing components
3. You need insight on how the modules interact with each other

## Architecture Overview

The Review Clusterer is organized with a clean separation of concerns using a controller-framework pattern:

1. **Controllers**: Serve as entry points for CLI commands, orchestrating business logic
2. **Framework**: Contains core functionality implementations separated by responsibility
3. **cli.py**: CLI API definition

## Directory Structure

```
src/review_clusterer/
├── cli.py                  # CLI command definitions using Click
├── controllers/            # Controllers for different functionalities
└── framework/              # Core implementation modules
```

## Key Components

### CSV Processing
- `CsvProcessor` handles reading, validation, and cleaning of review data
- Supports expected columns format with review ID, title, details, and rating

### Embedding
- Abstract `Embedder` interface defines common embedding operations
- Two implementations available:
  - `VoyageEmbedder`: Uses external VoyageAI API for high-quality embeddings
  - `LocalEmbedder`: Provides lightweight local embeddings without API dependencies

### Vector Database
- `ChromaRepository` manages the ChromaDB integration
- Handles storing and querying embeddings with metadata
- Creates collections based on input CSV filenames

### Clustering
- Supports two clustering approaches:
  - K-means clustering with automatic optimal cluster detection
  - HDBSCAN clustering for more advanced, noise-resilient clustering
- Optional UMAP dimensionality reduction for improved clustering quality
- Calculates cluster centers and distances for sorting reviews by representativeness

### LLM Integration
- `LLMClient` provides OpenAI-compatible API access for text generation
- Supports both free-form completion and structured JSON output
- Used for analyzing review clusters (work in progress)

### Reporting
- `markdown_report.py` generates structured Markdown reports
- Shows clusters sorted by rating with most representative reviews
- Includes metadata like cluster size, average ratings, and distances

## CLI Commands

The application exposes several commands:

1. `csv-test`: Validates and displays CSV review data
2. `index`: Processes reviews, generates embeddings, and stores in ChromaDB
3. `search`: Interactive semantic search for finding similar reviews
4. `cluster`: Clusters reviews and displays or exports results
5. `plot-elbow`: Visualizes clustering quality metrics
6. `llm-test` and `llm-structured-test`: Test LLM integrations

Each command provides various options for configuration, including:
- Local vs. remote embedding
- Clustering algorithm selection
- HDBSCAN and UMAP parameter tuning
- Output format control

## Data Flow

1. CSV files are processed through `CsvProcessor`
2. Reviews are embedded using either local or VoyageAI embeddings
3. Embeddings are stored in ChromaDB via `ChromaRepository`
4. Clustering is performed using `clusterer.py` functions
5. Results are displayed in the console or formatted into Markdown reports
6. (In development) LLM analysis of clusters to extract insights