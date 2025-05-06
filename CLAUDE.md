# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Generation guidelines

Keep things simple. Just solve the problem in the simplest way asked. Don't add extra features.
Warn me if you can't make sense of something, or can't fetch a documentation I link to.

## Avoid unnecessary state

Avoid having unnecessary Class members in python. I commonly see generated files having a generated class that receives a few parameters in its constructor to simply use them in a function call. Something like:

```
foo = Foo(arg1, arg2)
foo.bar()
```

instead of simply

```
foo = bar(arg1, arg2)
# or
foo = Foo()
foo.bar(arg1, arg2)
```

Only keep local state if necessary, e.g. if you create a DB wrapper for a certain file, then keep that file as state. Otherwise just use pure functions and use Classes as just a way to group functions.

## Be concise with comments

Keep comments minimal and focused on non-obvious information:
- Don't add comments that restate what the code already shows
- Only document parameter ranges/constraints that aren't evident from the type hints
- Skip "Args:" and "Returns:" sections that just repeat parameter and return types 
- Don't add class docstrings that only restate the class name
- Focus on explaining "why" not "what" when a rationale is needed
- Keep CLI command docstrings short and direct 

## Baby steps in slices, not layers

Create functionality in slices instead of layers. I.e. don't create all CLI commands, then all Controller modules.
Instead, create one cli command for one functionality, then its controller, then required auxiliary modules. Add only the necessary functions to each so we can run one functionality end to end and make sure it works. Then we move on to the next functionality.

## Avoid unnecessary __init__.py files

Only create __init__.py files in the root package to make it importable. Don't add empty __init__.py files in subfolders unless they are specifically needed for imports.

## Keep a conversation with me

# Project Overview

This is a "Customer Review Analysis System" - a Python-based tool that processes customer reviews using embeddings to enable semantic search and clustering. This system helps businesses extract insights
from reviews through vector-based analysis of text content.

Core Features to Implement

1. Process and embed customer reviews from CSV files
   1.1. CSV files of reviews have the columns format `id,created_at,reviewer_name,date,review_title,review_details,review_rating,url`
2. Store embeddings in a vector database for semantic search
3. Cluster reviews to identify common themes and sentiments
4. Search reviews using semantic similarity
5. Visualize reviews with formatted output using Rich
6. Analyze clusters using LLMs to generate insights and summaries

# Technical Requirements

We do this interactively and you should stop and ask me if you think a different aproach would make more sense at any point (including this guidelines file).

## Project setup

(I will already have the setup done by myself. No need to do anything, but just know that this is it)

1. Python 3.12 using pyenv local
2. Use Poetry for dependency management and packaging
3. Run code with `poetry run <command>`

## Dependencies

- Use voyageai to create embeddings. Documentation at https://docs.voyageai.com/docs/api-key-and-installation . Load the api key from a local file which I will add after you generate the code.
- Use chromaDB for storing the embeddings. Documentation at https://docs.trychroma.com/reference/python
- Use openai and anthropic for LLM-based review analysis. API keys can be provided via environment variables or text files.
- Stop and tell me

# Overall implementation architecture

1. The entry point is a CLI module with the commands below. This file only cares about the command parsing. It will delegate logic to other files I will describe below.

- csv_test - Args: <csv_file_path> - Just loads the CSV file and prints the first 5 rows to check that it works
- index - Args: <csv_file_path> -
  1. Loads the CSV file
  2. create embeddings for it using voyageai api. Embeddings will use the format "title:$title\n$rating/5 stars rating,content:$content". The rest of the fields of the csv just go in metadata.
  3. The target chromadb file will be named after the csv file (without the extension). Delete existing chromadb file if exists. Save embeddings to it.
- search - Args: <csv_file_path> - load chromadb file matching csv file (without extension). Open interactive mode:
  1. receive query text
  2. embed query using voyageapi and search in chromadb. Show top 3 results in sumary form.
  3. go back to interactive mode
- cluster - Args: <csv_file_path> -
  1. Figure out what cluster count makes sense for the data. I remember seeing some way that you would plot something and check where they curve of the hockey stick was and that was the sweet spot.
  2. Run the clustering algorithm using the most widely used python libray.
  3. Display the list of clusters: Each cluster is printed as a "Cluster 1..$cluster_count, $review_count reviews. Mean distance $mean_dist. Avg rating: $avg_rating/5 \n$summary_of_5_most_central_reviews
- llm-test - Options: --provider, --model, --prompt, --api-key-file - Tests LLM integration with a simple prompt
- llm-structured-test - Options: --provider, --model, --prompt, --schema-file, --api-key-file - Tests LLM's ability to provide structured JSON output

2. Multiple Controller modules. One for each slice of functionality coming for the command line. Ideally each CLI command will delegate the orchestration of everything to its controller. E.g. `cluster` command should have a `clusterController` method or class (if state is necessary).

3. Auxiliary Modules to encapsulate functionality
   3.1. A CsvProcessor module wrapping csv loading and access functionality (it probably will be oo and have state)
   3.2 A Embedder module that deals with calling voyager api and creating embeddings
   3.3 A db repository module that wraps access to chromadb for a specific file (probably oo and have state)
   3.4 A Cluster module that will receive a query result from the DB and create a cluster structure for it
   3.5 A LLMClient module that provides a unified interface for working with different LLM providers (OpenAI and Anthropic)

4. Any thing I have missed here. We do this interactively and you should stop and ask me if you think a different aproach would make more sense.
