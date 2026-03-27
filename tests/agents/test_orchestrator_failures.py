from collections.abc import Awaitable, Callable
from typing import Any

import pytest

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
    def __init__(self, execute: Callable[[RuntimeDeps, Any], Awaitable[Any]]) -> None:
        self.execute = execute


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
        "researcher": StubCapability(execute=execute_researcher),
        "coder": StubCapability(execute=execute_coder),
        "reviewer": StubCapability(execute=execute_reviewer),
    }

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
        "researcher": StubCapability(execute=execute_researcher),
        "coder": StubCapability(execute=execute_coder),
        "reviewer": StubCapability(execute=execute_reviewer),
    }

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
        "researcher": StubCapability(execute=execute_researcher),
        "coder": StubCapability(execute=execute_coder),
        "reviewer": StubCapability(execute=execute_reviewer),
    }

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
        "researcher": StubCapability(execute=execute_researcher),
        "coder": StubCapability(execute=execute_coder),
        "reviewer": StubCapability(execute=execute_reviewer),
    }

    monkeypatch.setattr(
        "ai_orchestration.agents.orchestrator.get_worker_capability",
        lambda name: capabilities[name],
    )

    result = await execute_orchestrator(deps=_deps(max_steps=2, retry_budget=3), request=_request())

    assert calls == ["researcher", "coder"]
    assert result.status is TaskStatus.PARTIAL
    assert result.failures[-1].kind is FailureKind.ORCHESTRATION_LIMIT
