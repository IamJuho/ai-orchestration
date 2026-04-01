from dataclasses import dataclass
from typing import Any

import pytest

from ai_orchestration.agents import coder, researcher, reviewer
from ai_orchestration.contracts.coder import CoderRequest, CoderResponse
from ai_orchestration.contracts.common import TaskStatus
from ai_orchestration.contracts.researcher import ResearcherRequest, ResearcherResponse
from ai_orchestration.contracts.reviewer import ReviewerRequest, ReviewerResponse
from ai_orchestration.deps import RuntimeDeps
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.provider import OpenAIProviderAdapter


@dataclass
class StubRunResult:
    output: Any


class StubAgent:
    def __init__(self, output: Any) -> None:
        self._output = output

    async def run(self, **_: Any) -> StubRunResult:
        return StubRunResult(output=self._output)


def _deps() -> RuntimeDeps:
    settings = Settings.model_validate({"OPENAI_API_KEY": "test-key"})
    provider = OpenAIProviderAdapter(settings=settings)
    return RuntimeDeps(config=settings, provider=provider, provider_label=provider.provider_label)


@pytest.mark.asyncio
async def test_execute_researcher_returns_typed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        researcher,
        "researcher_agent",
        StubAgent(
            {
                "status": TaskStatus.SUCCESS,
                "artifacts": [{"artifact_id": "r-1", "content": "notes"}],
            }
        ),
    )

    result = await researcher.execute_researcher(
        _deps(),
        ResearcherRequest(run_id="run-1", task_id="research", objective="Research topic"),
    )

    assert isinstance(result, ResearcherResponse)
    assert result.status is TaskStatus.SUCCESS
    assert result.failure is None
    assert result.artifacts[0].kind == "research_notes"


@pytest.mark.asyncio
async def test_execute_coder_returns_typed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        coder,
        "coder_agent",
        StubAgent(
            {
                "status": TaskStatus.SUCCESS,
                "artifacts": [{"artifact_id": "c-1", "content": "plan"}],
            }
        ),
    )

    result = await coder.execute_coder(
        _deps(),
        CoderRequest(run_id="run-1", task_id="code", objective="Build feature"),
    )

    assert isinstance(result, CoderResponse)
    assert result.status is TaskStatus.SUCCESS
    assert result.failure is None
    assert result.artifacts[0].kind == "implementation_plan"


@pytest.mark.asyncio
async def test_execute_reviewer_returns_typed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reviewer,
        "reviewer_agent",
        StubAgent(
            {
                "status": TaskStatus.SUCCESS,
                "artifacts": [{"artifact_id": "v-1", "content": "approved"}],
            }
        ),
    )

    result = await reviewer.execute_reviewer(
        _deps(),
        ReviewerRequest(run_id="run-1", task_id="review", objective="Review plan"),
    )

    assert isinstance(result, ReviewerResponse)
    assert result.status is TaskStatus.SUCCESS
    assert result.failure is None
    assert result.artifacts[0].kind == "review_report"
