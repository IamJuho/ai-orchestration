from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from ai_orchestration.agents.base import OrchestrationContext, UnknownWorkerCapabilityError
from ai_orchestration.contracts.coder import CoderArtifact, CoderRequest, CoderResponse
from ai_orchestration.contracts.common import Failure, FailureKind, TaskStatus
from ai_orchestration.contracts.orchestrator import OrchestratorRequest
from ai_orchestration.contracts.researcher import (
    ResearcherArtifact,
    ResearcherRequest,
    ResearcherResponse,
)
from ai_orchestration.contracts.reviewer import ReviewerArtifact, ReviewerRequest, ReviewerResponse
from ai_orchestration.deps import RuntimeDeps
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.runner import run_orchestration
from ai_orchestration.state.memory import InMemoryStateStore


class StubCapability:
    def __init__(
        self,
        execute: Callable[[RuntimeDeps, Any], Awaitable[Any]],
        request_factory: Callable[[OrchestratorRequest, OrchestrationContext], Any],
        *,
        fallback_retryable: bool = False,
    ) -> None:
        self.execute = execute
        self.name = "stub"
        self.request_model = object
        self.response_model = object
        self.request_factory = request_factory
        self.fallback_retryable = fallback_retryable
        self.terminal_on_success = False
        self.success_summary: str | None = None
        self.failure_terminal_status = lambda context: TaskStatus.FAILED
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


def _settings(*, timeout_seconds: float = 1.0, retry_budget: int = 2) -> Settings:
    return Settings.model_validate(
        {
            "OPENAI_API_KEY": "test-key",
            "timeout_seconds": timeout_seconds,
            "retry_budget": retry_budget,
        }
    )


@pytest.mark.asyncio
async def test_runner_persists_successful_run_and_append_only_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewer_calls = 0

    async def execute_researcher(_: RuntimeDeps, __: ResearcherRequest) -> ResearcherResponse:
        return ResearcherResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[ResearcherArtifact(artifact_id="r-1", content="notes")],
        )

    async def execute_coder(_: RuntimeDeps, __: CoderRequest) -> CoderResponse:
        return CoderResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[CoderArtifact(artifact_id="c-1", content="plan")],
        )

    async def execute_reviewer(_: RuntimeDeps, __: ReviewerRequest) -> ReviewerResponse:
        nonlocal reviewer_calls
        reviewer_calls += 1
        if reviewer_calls == 1:
            return ReviewerResponse(
                status=TaskStatus.PARTIAL,
                failure=Failure(
                    kind=FailureKind.WORKER,
                    message="needs one retry",
                    retryable=True,
                ),
            )
        return ReviewerResponse(
            status=TaskStatus.SUCCESS,
            artifacts=[ReviewerArtifact(artifact_id="v-1", content="approved")],
        )

    capabilities: dict[str, StubCapability] = {
        "researcher": StubCapability(
            execute=execute_researcher, request_factory=_researcher_request
        ),
        "coder": StubCapability(execute=execute_coder, request_factory=_coder_request),
        "reviewer": StubCapability(
            execute=execute_reviewer,
            request_factory=_reviewer_request,
            fallback_retryable=True,
        ),
    }
    capabilities["reviewer"].terminal_on_success = True
    capabilities[
        "reviewer"
    ].success_summary = "Completed researcher -> coder -> reviewer sequence successfully."
    capabilities["reviewer"].non_retryable_summary = lambda _: (
        "Reviewer rejected candidate output with non-retryable failure."
    )
    capabilities["reviewer"].retry_exhausted_summary = lambda _: (
        "Reviewer rejected candidate output after retry budget exhausted."
    )

    monkeypatch.setattr(
        "ai_orchestration.runtime.runner.get_worker_capability",
        lambda name: capabilities[name],
    )

    store = InMemoryStateStore()
    result = await run_orchestration(
        "Ship runtime runner",
        config=_settings(),
        state_store=store,
        task_id="task-main",
        run_id_factory=lambda: "run-success",
    )

    assert result.run_id == "run-success"
    assert result.status is TaskStatus.SUCCESS
    assert result.exit_status.value == "success"
    assert result.exit_code == 0

    run_state = await store.get_run("run-success")
    assert run_state is not None
    assert run_state.task_statuses["task-main"] is TaskStatus.SUCCESS
    assert run_state.task_statuses["task-main:reviewer"] is TaskStatus.SUCCESS
    assert len(run_state.artifacts) == 3
    assert len(run_state.failures) == 1
    assert run_state.failures[0].kind is FailureKind.WORKER

    assert tuple(event.event_type for event in run_state.trace_events) == (
        "run.started",
        "worker.dispatched",
        "worker.completed",
        "worker.dispatched",
        "worker.completed",
        "worker.dispatched",
        "worker.completed",
        "worker.retry",
        "worker.dispatched",
        "worker.completed",
        "run.completed",
    )
    assert run_state.trace_events[0].payload["objective"] == "Ship runtime runner"
    assert run_state.trace_events[7].payload["worker_name"] == "reviewer"
    assert run_state.trace_events[7].payload["attempt"] == "1"


@pytest.mark.asyncio
async def test_runner_persists_failed_run_and_terminal_failure_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def execute_researcher(_: RuntimeDeps, __: ResearcherRequest) -> ResearcherResponse:
        return ResearcherResponse(
            status=TaskStatus.FAILED,
            failure=Failure(
                kind=FailureKind.VALIDATION,
                message="bad schema",
                retryable=False,
            ),
        )

    async def execute_coder(_: RuntimeDeps, __: CoderRequest) -> CoderResponse:
        raise AssertionError("Coder should not execute after non-retryable researcher failure")

    async def execute_reviewer(_: RuntimeDeps, __: ReviewerRequest) -> ReviewerResponse:
        raise AssertionError("Reviewer should not execute after non-retryable researcher failure")

    capabilities: dict[str, StubCapability] = {
        "researcher": StubCapability(
            execute=execute_researcher, request_factory=_researcher_request
        ),
        "coder": StubCapability(execute=execute_coder, request_factory=_coder_request),
        "reviewer": StubCapability(
            execute=execute_reviewer,
            request_factory=_reviewer_request,
            fallback_retryable=True,
        ),
    }
    capabilities["reviewer"].terminal_on_success = True
    capabilities[
        "reviewer"
    ].success_summary = "Completed researcher -> coder -> reviewer sequence successfully."
    capabilities["reviewer"].non_retryable_summary = lambda _: (
        "Reviewer rejected candidate output with non-retryable failure."
    )
    capabilities["reviewer"].retry_exhausted_summary = lambda _: (
        "Reviewer rejected candidate output after retry budget exhausted."
    )

    monkeypatch.setattr(
        "ai_orchestration.runtime.runner.get_worker_capability",
        lambda name: capabilities[name],
    )

    store = InMemoryStateStore()
    result = await run_orchestration(
        "Ship runtime runner",
        config=_settings(),
        state_store=store,
        task_id="task-main",
        run_id_factory=lambda: "run-failed",
    )

    assert result.run_id == "run-failed"
    assert result.status is TaskStatus.FAILED
    assert result.exit_status.value == "failed"
    assert result.exit_code == 1
    assert result.failures[-1].kind is FailureKind.VALIDATION

    run_state = await store.get_run("run-failed")
    assert run_state is not None
    assert run_state.task_statuses["task-main"] is TaskStatus.FAILED
    assert run_state.task_statuses["task-main:researcher"] is TaskStatus.FAILED
    assert tuple(event.event_type for event in run_state.trace_events) == (
        "run.started",
        "worker.dispatched",
        "worker.completed",
        "run.failed",
    )
    assert run_state.trace_events[-1].payload["failure_kind"] == "validation"
    assert run_state.trace_events[-1].payload["status"] == "failed"


@pytest.mark.asyncio
async def test_runner_classifies_registry_wiring_failure_as_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stub_execute_orchestrator(**_: Any) -> Any:
        raise UnknownWorkerCapabilityError(
            capability="planner",
            available_capabilities=("researcher", "coder", "reviewer"),
        )

    monkeypatch.setattr(
        "ai_orchestration.runtime.runner.execute_orchestrator",
        stub_execute_orchestrator,
    )

    store = InMemoryStateStore()
    result = await run_orchestration(
        "Ship runtime runner",
        config=_settings(),
        state_store=store,
        task_id="task-config-failure",
        run_id_factory=lambda: "run-config-failure",
    )

    assert result.status is TaskStatus.FAILED
    assert result.failures[-1].kind is FailureKind.CONFIG

    run_state = await store.get_run("run-config-failure")
    assert run_state is not None
    assert run_state.trace_events[-1].event_type == "run.failed"
    assert run_state.trace_events[-1].payload["failure_kind"] == "config"


@pytest.mark.asyncio
async def test_runner_classifies_unexpected_runtime_exception_as_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stub_execute_orchestrator(**_: Any) -> Any:
        raise RuntimeError("runtime exploded")

    monkeypatch.setattr(
        "ai_orchestration.runtime.runner.execute_orchestrator",
        stub_execute_orchestrator,
    )

    store = InMemoryStateStore()
    result = await run_orchestration(
        "Ship runtime runner",
        config=_settings(),
        state_store=store,
        task_id="task-runtime-failure",
        run_id_factory=lambda: "run-runtime-failure",
    )

    assert result.status is TaskStatus.FAILED
    assert result.failures[-1].kind is FailureKind.WORKER
