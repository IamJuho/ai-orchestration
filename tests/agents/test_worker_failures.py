from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai import AgentRunError

from ai_orchestration.agents import coder, researcher, reviewer
from ai_orchestration.contracts.coder import CoderRequest
from ai_orchestration.contracts.common import FailureKind, TaskStatus
from ai_orchestration.contracts.researcher import ResearcherRequest
from ai_orchestration.contracts.reviewer import ReviewerRequest
from ai_orchestration.deps import RuntimeDeps
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.provider import OpenAIProviderAdapter


@dataclass
class StubRunResult:
    output: Any


class StubAgentRaises:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def run(self, **_: Any) -> StubRunResult:
        raise self._error


class StubAgentOutput:
    def __init__(self, output: Any) -> None:
        self._output = output

    async def run(self, **_: Any) -> StubRunResult:
        return StubRunResult(output=self._output)


def _deps() -> RuntimeDeps:
    settings = Settings.model_validate({"OPENAI_API_KEY": "test-key"})
    provider = OpenAIProviderAdapter(settings=settings)
    return RuntimeDeps(config=settings, provider=provider, provider_label=provider.provider_label)


@pytest.mark.asyncio
async def test_execute_researcher_maps_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        researcher, "researcher_agent", StubAgentRaises(AgentRunError("provider down"))
    )

    result = await researcher.execute_researcher(
        _deps(),
        ResearcherRequest(run_id="run-1", task_id="research", objective="Research topic"),
    )

    assert result.status is TaskStatus.FAILED
    assert result.failure is not None
    assert result.failure.kind is FailureKind.PROVIDER
    assert result.failure.retryable is True


@pytest.mark.asyncio
async def test_execute_coder_maps_validation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        coder,
        "coder_agent",
        StubAgentOutput(
            {
                "status": TaskStatus.SUCCESS,
                "artifacts": [
                    {
                        "artifact_id": "bad-coder",
                        "kind": "review_report",
                        "producer": "reviewer",
                        "content": "wrong artifact",
                    }
                ],
            }
        ),
    )

    result = await coder.execute_coder(
        _deps(),
        CoderRequest(run_id="run-1", task_id="code", objective="Build feature"),
    )

    assert result.status is TaskStatus.FAILED
    assert result.failure is not None
    assert result.failure.kind is FailureKind.VALIDATION
    assert result.failure.retryable is False


@pytest.mark.asyncio
async def test_execute_reviewer_maps_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reviewer, "reviewer_agent", StubAgentRaises(AgentRunError("provider timeout"))
    )

    result = await reviewer.execute_reviewer(
        _deps(),
        ReviewerRequest(run_id="run-1", task_id="review", objective="Review plan"),
    )

    assert result.status is TaskStatus.FAILED
    assert result.failure is not None
    assert result.failure.kind is FailureKind.PROVIDER
    assert result.failure.retryable is True
