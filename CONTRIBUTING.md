# Contributing to Agentic AI Toolkit

Thank you for your interest in contributing! This document provides guidelines for contributing to the Agentic AI Toolkit.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/MHHamdan/agentic-ai-toolkit.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install development dependencies: `pip install -e ".[dev]"`
5. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions/classes (Google style)
- Maximum line length: 88 characters

## Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Tests should be fast and deterministic
- Mock external API calls

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentic_toolkit

# Run specific test file
pytest tests/test_agents.py
```

## Paper-Code Consistency

When contributing implementations of paper concepts:

1. Reference the specific paper section in docstrings
2. Update `docs/PAPER_CODE_MAPPING.md` with the mapping
3. Include equation numbers where applicable
4. Add examples demonstrating the concept

Example docstring:
```python
def calculate_cnsr(successes: int, total_tasks: int, total_cost: float) -> float:
    """Calculate Cost-Normalized Success Rate (CNSR).

    From Section XI of the paper (Equation 9):
        CNSR = Success Rate / Mean Cost per Task
    ...
    """
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Squash commits before merging

### PR Title Format

Use conventional commit format:
- `feat: Add new feature`
- `fix: Fix bug in X`
- `docs: Update documentation`
- `refactor: Refactor X component`
- `test: Add tests for Y`

## Issue Guidelines

When opening issues:

- Use issue templates when available
- Provide minimal reproducible examples for bugs
- Include Python version, OS, and package versions
- Search existing issues before creating new ones

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

- Open a discussion on GitHub
- Check existing documentation
- Review closed issues for common questions
