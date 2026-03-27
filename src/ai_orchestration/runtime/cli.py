from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from ai_orchestration.contracts.orchestrator import FinalResponse
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.runner import run_orchestration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-orchestration")
    parser.add_argument("--objective", required=True)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-steps", type=int)
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    update: dict[str, float | int] = {}
    if args.timeout_seconds is not None:
        update["timeout_seconds"] = args.timeout_seconds
    if args.max_steps is not None:
        update["max_steps"] = args.max_steps
    return Settings.model_validate({**settings.model_dump(mode="python"), **update})


def _format_validation_error(exc: ValidationError) -> str:
    fields = sorted({".".join(str(part) for part in err["loc"]) for err in exc.errors()})
    if not fields:
        return "Configuration validation failed."
    return f"Configuration validation failed for: {', '.join(fields)}"


def _to_final_response_payload_json(response: FinalResponse) -> str:
    return json.dumps(response.model_dump(mode="json"), separators=(",", ":"))


def _to_plain_text(response: FinalResponse) -> str:
    lines = [
        f"run_id: {response.run_id}",
        f"status: {response.status.value}",
        f"summary: {response.summary}",
        f"artifacts: {len(response.artifacts)}",
    ]
    for artifact in response.artifacts:
        lines.append(
            f"- [{artifact.kind}] {artifact.artifact_id} ({artifact.producer}): {artifact.content}"
        )
    if response.failures:
        lines.append(f"failures: {len(response.failures)}")
        for failure in response.failures:
            lines.append(f"- [{failure.kind.value}] {failure.message}")
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    try:
        settings = _settings_from_args(args)
    except ValidationError as exc:
        print(_format_validation_error(exc), file=sys.stderr)
        return 2

    result = await run_orchestration(args.objective, config=settings)
    final_response = FinalResponse.model_validate(result.model_dump())

    if args.json_output:
        print(_to_final_response_payload_json(final_response))
    else:
        print(_to_plain_text(final_response))
    return result.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
