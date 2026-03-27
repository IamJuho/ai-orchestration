from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from ai_orchestration.agents.orchestrator import execute_orchestrator
from ai_orchestration.contracts.coder import CoderArtifact, CoderRequest, CoderResponse
from ai_orchestration.contracts.common import TaskEnvelope, TaskStatus
from ai_orchestration.contracts.orchestrator import OrchestratorRequest
from ai_orchestration.contracts.researcher import (
    ResearcherArtifact,
    ResearcherRequest,
    ResearcherResponse,
)
from ai_orchestration.contracts.reviewer import ReviewerArtifact, ReviewerRequest, ReviewerResponse
from ai_orchestration.deps import RuntimeDeps
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.provider import OpenAIProviderAdapter


def _deps(*, max_steps: int = 6, retry_budget: int = 2) -> RuntimeDeps:
    settings = Settings.model_validate(
        {
            "OPENAI_API_KEY": "test-key",
            "max_steps": max_steps,
            "retry_budget": retry_budget,
        }
    )
    provider = OpenAIProviderAdapter(settings=settings)
    return RuntimeDeps(config=settings, provider=provider, provider_label=provider.provider_label)


class StubCapability:
    def __init__(self, execute: Callable[[RuntimeDeps, Any], Awaitable[Any]]) -> None:
        self.execute = execute


@pytest.mark.asyncio
async def test_orchestrator_runs_fixed_sequence_and_returns_typed_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def execute_researcher(_: RuntimeDeps, request: ResearcherRequest) -> ResearcherResponse:
        calls.append("researcher")
        assert request.objective == "Ship deterministic orchestrator"
        return ResearcherResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[ResearcherArtifact(artifact_id="r-1", content="research notes")],
        )

    async def execute_coder(_: RuntimeDeps, request: CoderRequest) -> CoderResponse:
        calls.append("coder")
        assert len(request.research_artifacts) == 1
        return CoderResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[CoderArtifact(artifact_id="c-1", content="implementation plan")],
        )

    async def execute_reviewer(_: RuntimeDeps, request: ReviewerRequest) -> ReviewerResponse:
        calls.append("reviewer")
        assert len(request.candidate_artifacts) == 1
        return ReviewerResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[ReviewerArtifact(artifact_id="v-1", content="approved")],
        )

    capabilities: dict[str, StubCapability] = {
        "researcher": StubCapability(execute=execute_researcher),
        "coder": StubCapability(execute=execute_coder),
        "reviewer": StubCapability(execute=execute_reviewer),
    }

    def fake_get_worker_capability(name: str) -> StubCapability:
        return capabilities[name]

    monkeypatch.setattr(
        "ai_orchestration.agents.orchestrator.get_worker_capability",
        fake_get_worker_capability,
    )

    result = await execute_orchestrator(
        deps=_deps(),
        request=OrchestratorRequest(
            envelope=TaskEnvelope(
                run_id="run-1",
                task_id="task-1",
                objective="Ship deterministic orchestrator",
            )
        ),
    )

    assert calls == ["researcher", "coder", "reviewer"]
    assert result.status is TaskStatus.SUCCESS
    assert result.failures == []
    assert [artifact.producer for artifact in result.artifacts] == [
        "researcher",
        "coder",
        "reviewer",
    ]
