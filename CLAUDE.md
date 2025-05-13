# Python Development Guidelines

## Environment
- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Run commands with `poetry run <command>`

## Coding Principles

### Prefer Pure Functions
Avoid unnecessary class state. Only keep state when truly needed (e.g., DB connections).

```python
# Prefer this:
result = process_data(arg1, arg2)
# Or this:
processor = Processor()
result = processor.process(arg1, arg2)

# Instead of:
processor = Processor(arg1, arg2)
result = processor.process()
```

### Minimal __init__.py Files
Only create __init__.py files in the root package. Skip them in subfolders unless needed for imports.

## Project Documentation

The project has documentation files in the ai_docs directory:

### Reference Documentation
- **ai_docs/specs/overview.md**: High-level project description and key technologies. This provides essential context about what the system does without overwhelming details.

- **ai_docs/specs/implementation_details.md**: Detailed architecture information. Reference this when:
  - Making structural changes to the codebase
  - Adding new features that need to integrate with existing components
  - Needing insight on module interactions

### Prompt Templates
- **ai_docs/prompts/**: Contains LLM prompt templates used in the application