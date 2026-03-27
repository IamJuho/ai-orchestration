# AI Orchestration PoC Project

Requires Python 3.11.

## Install

```bash
uv sync --extra dev
```

## Environment setup

The runtime loads configuration from environment variables.

```bash
export OPENAI_API_KEY="your-api-key"
```

Optional overrides:

```bash
export AI_ORCHESTRATION_MODEL="openai:gpt-4o-mini"
export AI_ORCHESTRATION_TIMEOUT_SECONDS="30"
export AI_ORCHESTRATION_MAX_STEPS="6"
export AI_ORCHESTRATION_RETRY_BUDGET="2"
```

`OPENAI_API_KEY` is required for real CLI runs. The offline test suite stubs the provider and does not need a live key.

## CLI usage

Show help:

```bash
uv run --extra dev python -m ai_orchestration.runtime.cli --help
```

Run the CLI in plain text mode:

```bash
uv run --extra dev python -m ai_orchestration.runtime.cli \
  --objective "Draft a rollout plan" \
  --timeout-seconds 30 \
  --max-steps 6
```

Run the CLI in JSON mode:

```bash
uv run --extra dev python -m ai_orchestration.runtime.cli \
  --objective "Draft a rollout plan" \
  --json \
  --timeout-seconds 30 \
  --max-steps 6
```

Supported flags:

- `--objective`, required task description
- `--json`, emit the final response as compact JSON
- `--timeout-seconds`, override the runtime timeout
- `--max-steps`, override the orchestration step budget

## Verification

These are the local checks mirrored by CI:

```bash
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .
uv run --extra dev mypy src tests
uv run --extra dev pytest -m "not live"
```

`pytest` already defaults to `-m 'not live'` in `pyproject.toml`, but the explicit marker keeps the local command and CI command aligned.

## Offline tests

Run the offline suite only:

```bash
uv run --extra dev pytest -m "not live"
```

## Live smoke test

Live smoke tests are opt-in. They require `OPENAI_API_KEY` and are not part of CI.

Run the gated live pytest smoke test:

```bash
uv run --extra dev pytest -m live tests/runtime/test_cli_live.py
```

Run the equivalent CLI smoke command directly:

```bash
uv run --extra dev python -m ai_orchestration.runtime.cli \
  --objective "Live smoke verification" \
  --json \
  --max-steps 1 \
  --timeout-seconds 45
```
