import pytest

from ai_orchestration.contracts.common import Failure, FailureKind, TaskStatus
from ai_orchestration.runtime.runner import ExitStatus, RunnerResult


def test_cli_invalid_timeout_override_returns_exit_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from ai_orchestration.runtime.cli import main

    exit_code = main(["--objective", "invalid override", "--timeout-seconds", "0"])

    assert exit_code == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "Configuration validation failed for: timeout_seconds" in out.err


def test_cli_failed_runner_result_returns_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def stub_run_orchestration(objective: str, *, config: object) -> RunnerResult:
        assert objective == "runner failed"
        return RunnerResult(
            run_id="run-failed",
            status=TaskStatus.FAILED,
            summary="runtime failure",
            artifacts=[],
            failures=[Failure(kind=FailureKind.TIMEOUT, message="timed out")],
            exit_status=ExitStatus.FAILED,
            exit_code=1,
        )

    monkeypatch.setattr(
        "ai_orchestration.runtime.cli.run_orchestration",
        stub_run_orchestration,
    )

    from ai_orchestration.runtime.cli import main

    exit_code = main(["--objective", "runner failed", "--json"])

    assert exit_code == 1
    out = capsys.readouterr()
    assert out.err == ""
    assert '"status":"failed"' in out.out
    assert '"kind":"timeout"' in out.out
