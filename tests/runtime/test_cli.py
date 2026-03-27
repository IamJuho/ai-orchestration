import json

import pytest

from ai_orchestration.contracts.common import Artifact, Failure, FailureKind, TaskStatus
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.runner import ExitStatus, RunnerResult


def test_cli_json_mode_emits_final_response_shape(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def stub_run_orchestration(objective: str, *, config: Settings) -> RunnerResult:
        assert objective == "Ship CLI"
        assert config.timeout_seconds == 4.5
        assert config.max_steps == 8
        return RunnerResult(
            run_id="run-123",
            status=TaskStatus.SUCCESS,
            summary="completed",
            artifacts=[
                Artifact(
                    artifact_id="a-1",
                    kind="final_answer",
                    producer="orchestrator",
                    content="result",
                )
            ],
            failures=[],
            exit_status=ExitStatus.SUCCESS,
            exit_code=0,
        )

    monkeypatch.setattr(
        "ai_orchestration.runtime.cli.run_orchestration",
        stub_run_orchestration,
    )

    from ai_orchestration.runtime.cli import main

    exit_code = main(
        ["--objective", "Ship CLI", "--json", "--timeout-seconds", "4.5", "--max-steps", "8"]
    )

    assert exit_code == 0
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert out.err == ""
    assert set(payload) == {"run_id", "status", "summary", "artifacts", "failures"}
    assert payload["run_id"] == "run-123"
    assert payload["status"] == "success"
    assert payload["summary"] == "completed"
    assert payload["artifacts"][0]["kind"] == "final_answer"
    assert payload["failures"] == []


def test_cli_plain_text_mode_summarizes_status_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def stub_run_orchestration(objective: str, *, config: Settings) -> RunnerResult:
        assert objective == "Ship CLI text"
        assert config.timeout_seconds == 30.0
        assert config.max_steps == 6
        return RunnerResult(
            run_id="run-plain",
            status=TaskStatus.PARTIAL,
            summary="review requires follow-up",
            artifacts=[
                Artifact(
                    artifact_id="report-1",
                    kind="review_report",
                    producer="reviewer",
                    content="needs one more iteration",
                )
            ],
            failures=[
                Failure(
                    kind=FailureKind.WORKER,
                    message="review rejected",
                    retryable=False,
                )
            ],
            exit_status=ExitStatus.PARTIAL,
            exit_code=1,
        )

    monkeypatch.setattr(
        "ai_orchestration.runtime.cli.run_orchestration",
        stub_run_orchestration,
    )

    from ai_orchestration.runtime.cli import main

    exit_code = main(["--objective", "Ship CLI text"])

    assert exit_code == 1
    out = capsys.readouterr()
    assert out.err == ""
    assert "run_id: run-plain" in out.out
    assert "status: partial" in out.out
    assert "summary: review requires follow-up" in out.out
    assert "artifacts: 1" in out.out
    assert "[review_report] report-1 (reviewer): needs one more iteration" in out.out
    assert "failures: 1" in out.out
    assert "[worker] review rejected" in out.out
