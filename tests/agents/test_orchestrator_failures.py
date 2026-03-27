from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from ai_orchestration.agents.base import OrchestrationContext
from ai_orchestration.agents.orchestrator import execute_orchestrator
from ai_orchestration.contracts.coder import CoderArtifact, CoderRequest, CoderResponse
from ai_orchestration.contracts.common import Failure, FailureKind, TaskEnvelope, TaskStatus
from ai_orchestration.contracts.orchestrator import OrchestratorRequest
from ai_orchestration.contracts.researcher import (
    ResearcherArtifact,
    ResearcherRequest,
    ResearcherResponse,
)
from ai_orchestration.contracts.reviewer import ReviewerRequest, ReviewerResponse
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


def _request() -> OrchestratorRequest:
    return OrchestratorRequest(
        envelope=TaskEnvelope(
            run_id="run-1",
            task_id="task-1",
            objective="Stabilize orchestration failure handling",
        )
    )


class StubCapability:
    def __init__(
        self,
        execute: Callable[[RuntimeDeps, Any], Awaitable[Any]],
        request_factory: Callable[[OrchestratorRequest, OrchestrationContext], Any],
        *,
        fallback_retryable: bool = False,
        failure_terminal_status: Callable[[OrchestrationContext], TaskStatus] | None = None,
    ) -> None:
        self.execute = execute
        self.request_factory = request_factory
        self.fallback_retryable = fallback_retryable
        self.terminal_on_success = False
        self.success_summary: str | None = None
        self.failure_terminal_status = failure_terminal_status or (
            lambda context: TaskStatus.FAILED
        )
        self.max_steps_summary = lambda name: (
            f"Orchestrator exceeded max_steps before {name} completed."
        )
        self.non_retryable_summary = lambda name: (
            f"{name.capitalize()} failed with non-retryable error."
        )
        self.retry_exhausted_summary = lambda name: f"{name.capitalize()} retry budget exhausted."


def _researcher_request(req: OrchestratorRequest, _: OrchestrationContext) -> ResearcherRequest:
    return ResearcherRequest(
        run_id=req.envelope.run_id,
        task_id=req.envelope.task_id,
        objective=req.envelope.objective,
        constraints=req.envelope.constraints,
    )


def _coder_request(req: OrchestratorRequest, ctx: OrchestrationContext) -> CoderRequest:
    return CoderRequest(
        run_id=req.envelope.run_id,
        task_id=req.envelope.task_id,
        objective=req.envelope.objective,
        research_artifacts=[
            ResearcherArtifact.model_validate(artifact.model_dump())
            for artifact in ctx.artifacts_by_worker.get("researcher", [])
        ],
        constraints=req.envelope.constraints,
    )


def _reviewer_request(req: OrchestratorRequest, ctx: OrchestrationContext) -> ReviewerRequest:
    return ReviewerRequest(
        run_id=req.envelope.run_id,
        task_id=req.envelope.task_id,
        objective=req.envelope.objective,
        candidate_artifacts=[
            CoderArtifact.model_validate(artifact.model_dump())
            for artifact in ctx.artifacts_by_worker.get("coder", [])
        ],
        constraints=req.envelope.constraints,
    )


def _reviewer_terminal_status(context: OrchestrationContext) -> TaskStatus:
    return TaskStatus.PARTIAL if context.artifacts_by_worker.get("coder") else TaskStatus.FAILED


@pytest.mark.asyncio
async def test_orchestrator_retries_retryable_worker_failure_until_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def execute_researcher(_: RuntimeDeps, __: ResearcherRequest) -> ResearcherResponse:
        calls.append("researcher")
        return ResearcherResponse(
            status=TaskStatus.FAILED,
            failure=Failure(
                kind=FailureKind.PROVIDER,
                message="provider unavailable",
                retryable=True,
            ),
        )

    async def execute_coder(_: RuntimeDeps, __: CoderRequest) -> CoderResponse:
        raise AssertionError("Coder should not run when researcher retries are exhausted")

    async def execute_reviewer(_: RuntimeDeps, __: ReviewerRequest) -> ReviewerResponse:
        raise AssertionError("Reviewer should not run when researcher retries are exhausted")

    capabilities: dict[str, StubCapability] = {
        "researcher": StubCapability(
            execute=execute_researcher, request_factory=_researcher_request
        ),
        "coder": StubCapability(execute=execute_coder, request_factory=_coder_request),
        "reviewer": StubCapability(
            execute=execute_reviewer,
            request_factory=_reviewer_request,
            fallback_retryable=True,
            failure_terminal_status=_reviewer_terminal_status,
        ),
    }
    capabilities["reviewer"].non_retryable_summary = lambda _: (
        "Reviewer rejected candidate output with non-retryable failure."
    )
    capabilities["reviewer"].retry_exhausted_summary = lambda _: (
        "Reviewer rejected candidate output after retry budget exhausted."
    )

    monkeypatch.setattr(
        "ai_orchestration.agents.orchestrator.get_worker_capability",
        lambda name: capabilities[name],
    )

    result = await execute_orchestrator(deps=_deps(retry_budget=2), request=_request())

    assert calls == ["researcher", "researcher", "researcher"]
    assert result.status is TaskStatus.FAILED
    assert result.failures[-1].kind is FailureKind.PROVIDER


@pytest.mark.asyncio
async def test_orchestrator_fails_fast_for_non_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def execute_researcher(_: RuntimeDeps, __: ResearcherRequest) -> ResearcherResponse:
        calls.append("researcher")
        return ResearcherResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[ResearcherArtifact(artifact_id="r-1", content="notes")],
        )

    async def execute_coder(_: RuntimeDeps, __: CoderRequest) -> CoderResponse:
        calls.append("coder")
        return CoderResponse(
            status=TaskStatus.FAILED,
            failure=Failure(
                kind=FailureKind.VALIDATION,
                message="schema mismatch",
                retryable=False,
            ),
        )

    async def execute_reviewer(_: RuntimeDeps, __: ReviewerRequest) -> ReviewerResponse:
        calls.append("reviewer")
        return ReviewerResponse(status=TaskStatus.SUCCESS)

    capabilities: dict[str, StubCapability] = {
        "researcher": StubCapability(
            execute=execute_researcher, request_factory=_researcher_request
        ),
        "coder": StubCapability(execute=execute_coder, request_factory=_coder_request),
        "reviewer": StubCapability(
            execute=execute_reviewer,
            request_factory=_reviewer_request,
            fallback_retryable=True,
            failure_terminal_status=_reviewer_terminal_status,
        ),
    }
    capabilities["reviewer"].non_retryable_summary = lambda _: (
        "Reviewer rejected candidate output with non-retryable failure."
    )
    capabilities["reviewer"].retry_exhausted_summary = lambda _: (
        "Reviewer rejected candidate output after retry budget exhausted."
    )

    monkeypatch.setattr(
        "ai_orchestration.agents.orchestrator.get_worker_capability",
        lambda name: capabilities[name],
    )

    result = await execute_orchestrator(deps=_deps(retry_budget=3), request=_request())

    assert calls == ["researcher", "coder"]
    assert result.status is TaskStatus.FAILED
    assert result.failures[-1].kind is FailureKind.VALIDATION


@pytest.mark.asyncio
async def test_orchestrator_reviewer_rejection_becomes_partial_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def execute_researcher(_: RuntimeDeps, __: ResearcherRequest) -> ResearcherResponse:
        calls.append("researcher")
        return ResearcherResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[ResearcherArtifact(artifact_id="r-1", content="notes")],
        )

    async def execute_coder(_: RuntimeDeps, __: CoderRequest) -> CoderResponse:
        calls.append("coder")
        return CoderResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[CoderArtifact(artifact_id="c-1", content="plan")],
        )

    async def execute_reviewer(_: RuntimeDeps, __: ReviewerRequest) -> ReviewerResponse:
        calls.append("reviewer")
        return ReviewerResponse(status=TaskStatus.PARTIAL)

    capabilities: dict[str, StubCapability] = {
        "researcher": StubCapability(
            execute=execute_researcher, request_factory=_researcher_request
        ),
        "coder": StubCapability(execute=execute_coder, request_factory=_coder_request),
        "reviewer": StubCapability(
            execute=execute_reviewer,
            request_factory=_reviewer_request,
            fallback_retryable=True,
            failure_terminal_status=_reviewer_terminal_status,
        ),
    }
    capabilities["reviewer"].non_retryable_summary = lambda _: (
        "Reviewer rejected candidate output with non-retryable failure."
    )
    capabilities["reviewer"].retry_exhausted_summary = lambda _: (
        "Reviewer rejected candidate output after retry budget exhausted."
    )

    monkeypatch.setattr(
        "ai_orchestration.agents.orchestrator.get_worker_capability",
        lambda name: capabilities[name],
    )

    result = await execute_orchestrator(deps=_deps(retry_budget=1), request=_request())

    assert calls == ["researcher", "coder", "reviewer", "reviewer"]
    assert result.status is TaskStatus.PARTIAL
    assert result.failures[-1].kind is FailureKind.WORKER


@pytest.mark.asyncio
async def test_orchestrator_stops_when_max_steps_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def execute_researcher(_: RuntimeDeps, __: ResearcherRequest) -> ResearcherResponse:
        calls.append("researcher")
        return ResearcherResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[ResearcherArtifact(artifact_id="r-1", content="notes")],
        )

    async def execute_coder(_: RuntimeDeps, __: CoderRequest) -> CoderResponse:
        calls.append("coder")
        return CoderResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[CoderArtifact(artifact_id="c-1", content="plan")],
        )

    async def execute_reviewer(_: RuntimeDeps, __: ReviewerRequest) -> ReviewerResponse:
        calls.append("reviewer")
        return ReviewerResponse(status=TaskStatus.SUCCESS)

    capabilities: dict[str, StubCapability] = {
        "researcher": StubCapability(
            execute=execute_researcher, request_factory=_researcher_request
        ),
        "coder": StubCapability(execute=execute_coder, request_factory=_coder_request),
        "reviewer": StubCapability(
            execute=execute_reviewer,
            request_factory=_reviewer_request,
            fallback_retryable=True,
            failure_terminal_status=_reviewer_terminal_status,
        ),
    }
    capabilities["reviewer"].non_retryable_summary = lambda _: (
        "Reviewer rejected candidate output with non-retryable failure."
    )
    capabilities["reviewer"].retry_exhausted_summary = lambda _: (
        "Reviewer rejected candidate output after retry budget exhausted."
    )

    monkeypatch.setattr(
        "ai_orchestration.agents.orchestrator.get_worker_capability",
        lambda name: capabilities[name],
    )

    result = await execute_orchestrator(deps=_deps(max_steps=2, retry_budget=3), request=_request())

    assert calls == ["researcher", "coder"]
    assert result.status is TaskStatus.PARTIAL
    assert result.failures[-1].kind is FailureKind.ORCHESTRATION_LIMIT
