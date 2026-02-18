# CQEDToolbox

Implementation of Tools For Experiments used in Pfafflab

## Installation

### For development:

```bash
# Using pip
pip install -e ".[dev]"

# Using uv
uv pip install -e ".[dev]"

# Using conda (after creating/activating environment)
pip install -e ".[dev]"
```

### For production:

```bash
pip install cqedtoolbox
```

## Development

### Setup

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Install pre-commit hooks: `pre-commit install`

### Running tests

```bash
pytest
```

### Code quality

```bash
# Format and lint
ruff check --fix .
ruff format .

# Type checking
mypy .
```

## License

MIT License - see LICENSE file for details
