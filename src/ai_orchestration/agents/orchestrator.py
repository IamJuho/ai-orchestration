from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ai_orchestration.agents.base import WorkerCapability
from ai_orchestration.agents.registry import CapabilityName, get_worker_capability
from ai_orchestration.contracts.coder import CoderArtifact, CoderRequest, CoderResponse
from ai_orchestration.contracts.common import Artifact, Failure, FailureKind, TaskStatus
from ai_orchestration.contracts.orchestrator import FinalResponse, OrchestratorRequest
from ai_orchestration.contracts.researcher import (
    ResearcherArtifact,
    ResearcherRequest,
    ResearcherResponse,
)
from ai_orchestration.contracts.reviewer import ReviewerRequest, ReviewerResponse
from ai_orchestration.deps import RuntimeDeps

CapabilityResolver = Callable[[str], WorkerCapability[Any, Any]]
RetryNotifier = Callable[[str, int, Failure], Awaitable[None]]
WorkerRequest = ResearcherRequest | CoderRequest | ReviewerRequest
WorkerResponse = ResearcherResponse | CoderResponse | ReviewerResponse


@dataclass(frozen=True)
class WorkerStep:
    worker_name: CapabilityName
    build_request: Callable[[OrchestratorRequest, OrchestratorState], WorkerRequest]


@dataclass
class OrchestratorState:
    steps_used: int = 0
    artifacts: list[Artifact] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    researcher_artifacts: list[ResearcherArtifact] = field(default_factory=list)
    coder_artifacts: list[CoderArtifact] = field(default_factory=list)


def _build_researcher_request(
    request: OrchestratorRequest, state: OrchestratorState
) -> ResearcherRequest:
    del state
    envelope = request.envelope
    return ResearcherRequest(
        run_id=envelope.run_id,
        task_id=envelope.task_id,
        objective=envelope.objective,
        constraints=envelope.constraints,
    )


def _build_coder_request(request: OrchestratorRequest, state: OrchestratorState) -> CoderRequest:
    envelope = request.envelope
    return CoderRequest(
        run_id=envelope.run_id,
        task_id=envelope.task_id,
        objective=envelope.objective,
        research_artifacts=state.researcher_artifacts,
        constraints=envelope.constraints,
    )


def _build_reviewer_request(
    request: OrchestratorRequest, state: OrchestratorState
) -> ReviewerRequest:
    envelope = request.envelope
    return ReviewerRequest(
        run_id=envelope.run_id,
        task_id=envelope.task_id,
        objective=envelope.objective,
        candidate_artifacts=state.coder_artifacts,
        constraints=envelope.constraints,
    )


def _v1_worker_steps() -> tuple[WorkerStep, WorkerStep, WorkerStep]:
    return (
        WorkerStep(worker_name="researcher", build_request=_build_researcher_request),
        WorkerStep(worker_name="coder", build_request=_build_coder_request),
        WorkerStep(worker_name="reviewer", build_request=_build_reviewer_request),
    )


def _to_common_artifacts(
    worker_artifacts: Sequence[BaseModel | Mapping[str, object]],
) -> list[Artifact]:
    common_artifacts: list[Artifact] = []
    for artifact in worker_artifacts:
        payload = artifact.model_dump() if isinstance(artifact, BaseModel) else dict(artifact)
        common_artifacts.append(Artifact.model_validate(payload))
    return common_artifacts


def _fallback_failure(worker_name: str, status: TaskStatus, retryable: bool) -> Failure:
    return Failure(
        kind=FailureKind.WORKER,
        message=f"{worker_name} returned {status.value} without failure details.",
        retryable=retryable,
        details={"worker": worker_name},
    )


def _max_steps_failure(worker_name: str, max_steps: int) -> Failure:
    return Failure(
        kind=FailureKind.ORCHESTRATION_LIMIT,
        message="Orchestrator reached max_steps before completing worker sequence.",
        details={"worker": worker_name, "max_steps": str(max_steps)},
    )


def _failed_response(
    run_id: str, summary: str, artifacts: list[Artifact], failures: list[Failure]
) -> FinalResponse:
    return FinalResponse(
        run_id=run_id,
        status=TaskStatus.FAILED,
        summary=summary,
        artifacts=artifacts,
        failures=failures,
    )


def _review_terminal_status(candidate_artifacts: list[CoderArtifact]) -> TaskStatus:
    return TaskStatus.PARTIAL if candidate_artifacts else TaskStatus.FAILED


def _max_steps_response(
    step: WorkerStep,
    *,
    request: OrchestratorRequest,
    state: OrchestratorState,
    max_steps: int,
) -> FinalResponse:
    limit_failure = _max_steps_failure(step.worker_name, max_steps)
    if step.worker_name == "reviewer":
        return FinalResponse(
            run_id=request.envelope.run_id,
            status=_review_terminal_status(state.coder_artifacts),
            summary="Orchestrator exceeded max_steps before reviewer completed.",
            artifacts=state.artifacts,
            failures=[*state.failures, limit_failure],
        )

    return _failed_response(
        run_id=request.envelope.run_id,
        summary=f"Orchestrator exceeded max_steps before {step.worker_name} completed.",
        artifacts=state.artifacts,
        failures=[*state.failures, limit_failure],
    )


def _non_retryable_response(
    step: WorkerStep,
    *,
    request: OrchestratorRequest,
    state: OrchestratorState,
) -> FinalResponse:
    if step.worker_name == "reviewer":
        return FinalResponse(
            run_id=request.envelope.run_id,
            status=_review_terminal_status(state.coder_artifacts),
            summary="Reviewer rejected candidate output with non-retryable failure.",
            artifacts=state.artifacts,
            failures=state.failures,
        )

    return _failed_response(
        run_id=request.envelope.run_id,
        summary=f"{step.worker_name.capitalize()} failed with non-retryable error.",
        artifacts=state.artifacts,
        failures=state.failures,
    )


def _retry_exhausted_response(
    step: WorkerStep,
    *,
    request: OrchestratorRequest,
    state: OrchestratorState,
) -> FinalResponse:
    if step.worker_name == "reviewer":
        return FinalResponse(
            run_id=request.envelope.run_id,
            status=_review_terminal_status(state.coder_artifacts),
            summary="Reviewer rejected candidate output after retry budget exhausted.",
            artifacts=state.artifacts,
            failures=state.failures,
        )

    return _failed_response(
        run_id=request.envelope.run_id,
        summary=f"{step.worker_name.capitalize()} retry budget exhausted.",
        artifacts=state.artifacts,
        failures=state.failures,
    )


def _handle_step_success(
    step: WorkerStep,
    *,
    request: OrchestratorRequest,
    state: OrchestratorState,
    response: WorkerResponse,
) -> FinalResponse | None:
    if step.worker_name == "researcher":
        assert isinstance(response, ResearcherResponse)
        state.researcher_artifacts = list(response.artifacts)
        state.artifacts.extend(_to_common_artifacts(response.artifacts))
        return None

    if step.worker_name == "coder":
        assert isinstance(response, CoderResponse)
        state.coder_artifacts = list(response.artifacts)
        state.artifacts.extend(_to_common_artifacts(response.artifacts))
        return None

    assert isinstance(response, ReviewerResponse)
    state.artifacts.extend(_to_common_artifacts(response.artifacts))
    return FinalResponse(
        run_id=request.envelope.run_id,
        status=TaskStatus.SUCCESS,
        summary="Completed researcher -> coder -> reviewer sequence successfully.",
        artifacts=state.artifacts,
        failures=state.failures,
    )


async def _execute_worker_step(
    *,
    deps: RuntimeDeps,
    request: OrchestratorRequest,
    step: WorkerStep,
    state: OrchestratorState,
    retry_budget: int,
    max_steps: int,
    resolve_capability: CapabilityResolver,
    on_retry: RetryNotifier | None,
) -> FinalResponse | None:
    capability = resolve_capability(step.worker_name)
    worker_request = step.build_request(request, state)

    for attempt in range(retry_budget + 1):
        if state.steps_used >= max_steps:
            return _max_steps_response(step, request=request, state=state, max_steps=max_steps)

        state.steps_used += 1
        response = await capability.execute(deps, worker_request)

        if response.status is TaskStatus.SUCCESS:
            return _handle_step_success(step, request=request, state=state, response=response)

        failure = response.failure or _fallback_failure(
            worker_name=step.worker_name,
            status=response.status,
            retryable=step.worker_name == "reviewer",
        )
        state.failures.append(failure)

        if not failure.retryable:
            return _non_retryable_response(step, request=request, state=state)
        if attempt == retry_budget:
            return _retry_exhausted_response(step, request=request, state=state)
        if on_retry is not None:
            await on_retry(step.worker_name, attempt + 1, failure)

    return _failed_response(
        run_id=request.envelope.run_id,
        summary="Unexpected worker step termination.",
        artifacts=state.artifacts,
        failures=state.failures,
    )


async def execute_orchestrator(
    deps: RuntimeDeps,
    request: OrchestratorRequest,
    *,
    capability_resolver: CapabilityResolver | None = None,
    on_retry: RetryNotifier | None = None,
) -> FinalResponse:
    resolve_capability = capability_resolver or get_worker_capability
    state = OrchestratorState(artifacts=list(request.envelope.input_artifacts))

    for step in _v1_worker_steps():
        step_result = await _execute_worker_step(
            deps=deps,
            request=request,
            step=step,
            state=state,
            retry_budget=deps.config.retry_budget,
            max_steps=deps.config.max_steps,
            resolve_capability=resolve_capability,
            on_retry=on_retry,
        )
        if step_result is not None:
            return step_result

    return _failed_response(
        run_id=request.envelope.run_id,
        summary="Unexpected orchestrator termination.",
        artifacts=state.artifacts,
        failures=state.failures,
    )
