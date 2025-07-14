# Packaging and Import Rules

This document outlines key conventions and lessons learned to ensure consistent and error-free development within the `cs-master-agent` project.

## 1. Python Packaging and `pyproject.toml`

**Rule:** To ensure `langgraph-cli` correctly recognizes and loads the project as a proper package, the `[tool.setuptools]` configuration in `pyproject.toml` must follow a specific namespacing convention.

- Always define the package under both a `langgraph.templates.*` namespace and its direct name.
- Explicitly map these package names to the source directory using `[tool.setuptools.package-dir]`.

**Reasoning:** The `langgraph-cli` tool appears to have an implicit requirement for a `langgraph.templates.*` namespace to correctly identify the project structure, especially for relative imports to work as expected after installation. Failing to follow this convention leads to `ImportError: attempted relative import with no known parent package`.

**Example (`pyproject.toml`):**

```toml
[tool.setuptools]
packages = ["langgraph.templates.parsing_graph", "parsing_graph"]

[tool.setuptools.package-dir]
"langgraph.templates.parsing_graph" = "src/parsing_graph"
"parsing_graph" = "src/parsing_graph"
```

## 2. Absolute Imports within the Package

**Rule:** After the project is installed in editable mode (`pip install -e .`), all intra-package imports must use absolute paths starting from the package name.

- **Don't** use relative imports like: `from .state import ParsingState`
- **Do** use absolute imports like: `from parsing_graph.state import ParsingState`

**Reasoning:** Using absolute imports removes any ambiguity for the Python interpreter when locating modules. It makes it clear that the import is coming from within the installed `parsing_graph` package, rather than relying on the file's current location. This is the most robust way to prevent `ImportError`.
