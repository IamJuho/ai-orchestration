from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel, Field

from ai_orchestration.agents.base import OrchestrationContext
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
    def __init__(
        self,
        execute: Callable[[RuntimeDeps, Any], Awaitable[Any]],
        request_factory: Callable[[OrchestratorRequest, OrchestrationContext], Any],
        *,
        fallback_retryable: bool = False,
    ) -> None:
        self.execute = execute
        self.request_factory = request_factory
        self.fallback_retryable = fallback_retryable
        self.success_summary = lambda sequence: (
            f"Completed {' -> '.join(sequence)} sequence successfully."
        )
        self.failure_terminal_status = lambda context: TaskStatus.FAILED
        self.max_steps_summary = lambda name: (
            f"Orchestrator exceeded max_steps before {name} completed."
        )
        self.non_retryable_summary = lambda name: (
            f"{name.capitalize()} failed with non-retryable error."
        )
        self.retry_exhausted_summary = lambda name: f"{name.capitalize()} retry budget exhausted."


class PublisherArtifact(BaseModel):
    artifact_id: str
    kind: str = "publication_packet"
    producer: str = "publisher"
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class PublisherResponse(BaseModel):
    status: TaskStatus
    artifacts: list[PublisherArtifact] = Field(default_factory=list)


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
        "researcher": StubCapability(
            execute=execute_researcher,
            request_factory=lambda req, _: ResearcherRequest(
                run_id=req.envelope.run_id,
                task_id=req.envelope.task_id,
                objective=req.envelope.objective,
                constraints=req.envelope.constraints,
            ),
        ),
        "coder": StubCapability(
            execute=execute_coder,
            request_factory=lambda req, _: ResearcherRequest(
                run_id=req.envelope.run_id,
                task_id=req.envelope.task_id,
                objective=req.envelope.objective,
            ),
        ),
        "reviewer": StubCapability(
            execute=execute_reviewer,
            request_factory=lambda req, _: ResearcherRequest(
                run_id=req.envelope.run_id,
                task_id=req.envelope.task_id,
                objective=req.envelope.objective,
            ),
            fallback_retryable=True,
        ),
    }

    capabilities["researcher"].request_factory = lambda req, _: ResearcherRequest(
        run_id=req.envelope.run_id,
        task_id=req.envelope.task_id,
        objective=req.envelope.objective,
        constraints=req.envelope.constraints,
    )
    capabilities["coder"].request_factory = lambda req, ctx: CoderRequest(
        run_id=req.envelope.run_id,
        task_id=req.envelope.task_id,
        objective=req.envelope.objective,
        research_artifacts=[
            ResearcherArtifact.model_validate(artifact.model_dump())
            for artifact in ctx.artifacts_by_worker.get("researcher", [])
        ],
        constraints=req.envelope.constraints,
    )
    capabilities["reviewer"].request_factory = lambda req, ctx: ReviewerRequest(
        run_id=req.envelope.run_id,
        task_id=req.envelope.task_id,
        objective=req.envelope.objective,
        candidate_artifacts=[
            CoderArtifact.model_validate(artifact.model_dump())
            for artifact in ctx.artifacts_by_worker.get("coder", [])
        ],
        constraints=req.envelope.constraints,
    )

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


@pytest.mark.asyncio
async def test_orchestrator_supports_appending_future_worker_via_registry_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def execute_researcher(_: RuntimeDeps, request: ResearcherRequest) -> ResearcherResponse:
        calls.append("researcher")
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

    async def execute_publisher(_: RuntimeDeps, request: ReviewerRequest) -> PublisherResponse:
        calls.append("publisher")
        assert len(request.candidate_artifacts) == 1
        return PublisherResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[PublisherArtifact(artifact_id="p-1", content="published")],
        )

    capabilities: dict[str, StubCapability] = {
        "researcher": StubCapability(
            execute=execute_researcher,
            request_factory=lambda req, _: ResearcherRequest(
                run_id=req.envelope.run_id,
                task_id=req.envelope.task_id,
                objective=req.envelope.objective,
                constraints=req.envelope.constraints,
            ),
        ),
        "coder": StubCapability(
            execute=execute_coder,
            request_factory=lambda req, ctx: CoderRequest(
                run_id=req.envelope.run_id,
                task_id=req.envelope.task_id,
                objective=req.envelope.objective,
                research_artifacts=[
                    ResearcherArtifact.model_validate(artifact.model_dump())
                    for artifact in ctx.artifacts_by_worker.get("researcher", [])
                ],
                constraints=req.envelope.constraints,
            ),
        ),
        "reviewer": StubCapability(
            execute=execute_reviewer,
            request_factory=lambda req, ctx: ReviewerRequest(
                run_id=req.envelope.run_id,
                task_id=req.envelope.task_id,
                objective=req.envelope.objective,
                candidate_artifacts=[
                    CoderArtifact.model_validate(artifact.model_dump())
                    for artifact in ctx.artifacts_by_worker.get("coder", [])
                ],
                constraints=req.envelope.constraints,
            ),
            fallback_retryable=True,
        ),
        "publisher": StubCapability(
            execute=execute_publisher,
            request_factory=lambda req, ctx: ReviewerRequest(
                run_id=req.envelope.run_id,
                task_id=req.envelope.task_id,
                objective=req.envelope.objective,
                candidate_artifacts=[
                    CoderArtifact.model_validate(artifact.model_dump())
                    for artifact in ctx.artifacts_by_worker.get("coder", [])
                ],
                constraints=req.envelope.constraints,
            ),
        ),
    }

    monkeypatch.setattr(
        "ai_orchestration.agents.orchestrator.list_worker_capabilities",
        lambda: ("researcher", "coder", "reviewer", "publisher"),
    )
    monkeypatch.setattr(
        "ai_orchestration.agents.orchestrator.get_worker_capability",
        lambda name: capabilities[name],
    )

    result = await execute_orchestrator(
        deps=_deps(),
        request=OrchestratorRequest(
            envelope=TaskEnvelope(
                run_id="run-2",
                task_id="task-2",
                objective="Ship deterministic orchestrator",
            )
        ),
    )

    assert calls == ["researcher", "coder", "reviewer", "publisher"]
    assert result.status is TaskStatus.SUCCESS
    assert (
        result.summary
        == "Completed researcher -> coder -> reviewer -> publisher sequence successfully."
    )
    assert result.artifacts[-1].producer == "publisher"
    assert result.artifacts[-1].kind == "publication_packet"
