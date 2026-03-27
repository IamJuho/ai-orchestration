from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ai_orchestration.agents.base import OrchestrationContext, WorkerCapability
from ai_orchestration.agents.registry import (
    CapabilityName,
    get_worker_capability,
    list_worker_capabilities,
)
from ai_orchestration.contracts.common import Artifact, Failure, FailureKind, TaskStatus
from ai_orchestration.contracts.orchestrator import FinalResponse, OrchestratorRequest
from ai_orchestration.deps import RuntimeDeps

CapabilityResolver = Callable[[str], WorkerCapability[Any, Any]]
RetryNotifier = Callable[[str, int, Failure], Awaitable[None]]


@dataclass(frozen=True)
class WorkerStep:
    worker_name: CapabilityName


@dataclass
class OrchestratorState:
    steps_used: int = 0
    context: OrchestrationContext = field(default_factory=OrchestrationContext)


def _v1_worker_steps() -> tuple[WorkerStep, ...]:
    return tuple(WorkerStep(worker_name=name) for name in list_worker_capabilities())


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
    *,
    run_id: str,
    status: TaskStatus,
    summary: str,
    context: OrchestrationContext,
    extra_failures: Sequence[Failure] = (),
) -> FinalResponse:
    return FinalResponse(
        run_id=run_id,
        status=status,
        summary=summary,
        artifacts=context.artifacts,
        failures=[*context.failures, *extra_failures],
    )


def _max_steps_response(
    *,
    step: WorkerStep,
    capability: WorkerCapability[Any, Any],
    request: OrchestratorRequest,
    state: OrchestratorState,
    max_steps: int,
) -> FinalResponse:
    limit_failure = _max_steps_failure(step.worker_name, max_steps)
    terminal_status = capability.failure_terminal_status(state.context)
    return _failed_response(
        run_id=request.envelope.run_id,
        status=terminal_status,
        summary=capability.max_steps_summary(step.worker_name),
        context=state.context,
        extra_failures=(limit_failure,),
    )


def _non_retryable_response(
    *,
    step: WorkerStep,
    capability: WorkerCapability[Any, Any],
    request: OrchestratorRequest,
    state: OrchestratorState,
) -> FinalResponse:
    terminal_status = capability.failure_terminal_status(state.context)
    return _failed_response(
        run_id=request.envelope.run_id,
        status=terminal_status,
        summary=capability.non_retryable_summary(step.worker_name),
        context=state.context,
    )


def _retry_exhausted_response(
    *,
    step: WorkerStep,
    capability: WorkerCapability[Any, Any],
    request: OrchestratorRequest,
    state: OrchestratorState,
) -> FinalResponse:
    terminal_status = capability.failure_terminal_status(state.context)
    return _failed_response(
        run_id=request.envelope.run_id,
        status=terminal_status,
        summary=capability.retry_exhausted_summary(step.worker_name),
        context=state.context,
    )


def _handle_step_success(
    *,
    step: WorkerStep,
    capability: WorkerCapability[Any, Any],
    request: OrchestratorRequest,
    state: OrchestratorState,
    response: BaseModel,
) -> FinalResponse | None:
    artifacts = _to_common_artifacts(getattr(response, "artifacts", []))
    state.context.artifacts.extend(artifacts)
    state.context.artifacts_by_worker[step.worker_name] = artifacts

    if not capability.terminal_on_success:
        return None

    return FinalResponse(
        run_id=request.envelope.run_id,
        status=TaskStatus.SUCCESS,
        summary=capability.success_summary or f"Completed {step.worker_name} successfully.",
        artifacts=state.context.artifacts,
        failures=state.context.failures,
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

    for attempt in range(retry_budget + 1):
        if state.steps_used >= max_steps:
            return _max_steps_response(
                step=step,
                capability=capability,
                request=request,
                state=state,
                max_steps=max_steps,
            )

        state.steps_used += 1
        worker_request = capability.request_factory(request, state.context)
        response = await capability.execute(deps, worker_request)

        if response.status is TaskStatus.SUCCESS:
            return _handle_step_success(
                step=step,
                capability=capability,
                request=request,
                state=state,
                response=response,
            )

        failure = response.failure or _fallback_failure(
            worker_name=step.worker_name,
            status=response.status,
            retryable=capability.fallback_retryable,
        )
        state.context.failures.append(failure)

        if not failure.retryable:
            return _non_retryable_response(
                step=step,
                capability=capability,
                request=request,
                state=state,
            )
        if attempt == retry_budget:
            return _retry_exhausted_response(
                step=step,
                capability=capability,
                request=request,
                state=state,
            )
        if on_retry is not None:
            await on_retry(step.worker_name, attempt + 1, failure)

    return _failed_response(
        run_id=request.envelope.run_id,
        status=TaskStatus.FAILED,
        summary="Unexpected worker step termination.",
        context=state.context,
    )


async def execute_orchestrator(
    deps: RuntimeDeps,
    request: OrchestratorRequest,
    *,
    capability_resolver: CapabilityResolver | None = None,
    on_retry: RetryNotifier | None = None,
) -> FinalResponse:
    resolve_capability = capability_resolver or get_worker_capability
    state = OrchestratorState(
        context=OrchestrationContext(artifacts=list(request.envelope.input_artifacts))
    )

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
        status=TaskStatus.FAILED,
        summary="Unexpected orchestrator termination.",
        context=state.context,
    )
