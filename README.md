# AI Orchestration PoC Project

## Bootstrap

Requires Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
```

## Local verification

```bash
ruff format --check .
ruff check .
mypy src tests
pytest
```
