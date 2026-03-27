from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from ai_orchestration.agents.base import UnknownWorkerCapabilityError, WorkerCapability
from ai_orchestration.agents.orchestrator import execute_orchestrator
from ai_orchestration.agents.registry import get_worker_capability
from ai_orchestration.contracts.common import Failure, FailureKind, TaskEnvelope, TaskStatus
from ai_orchestration.contracts.orchestrator import FinalResponse, OrchestratorRequest
from ai_orchestration.deps import RuntimeDeps, build_runtime_deps
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.tracing import (
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    WorkerCompletedEvent,
    WorkerDispatchedEvent,
    WorkerRetryEvent,
    to_trace_event,
)
from ai_orchestration.state.base import StateStore
from ai_orchestration.state.memory import InMemoryStateStore

DEFAULT_ROOT_TASK_ID = "orchestration"


class ExitStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class RunnerResult(FinalResponse):
    exit_status: ExitStatus
    exit_code: int


def create_run_id() -> str:
    return f"run-{uuid4().hex}"


def normalize_terminal_status(status: TaskStatus) -> TaskStatus:
    if status in (TaskStatus.SUCCESS, TaskStatus.PARTIAL, TaskStatus.FAILED):
        return status
    return TaskStatus.FAILED


def _to_exit_status(status: TaskStatus) -> ExitStatus:
    if status is TaskStatus.SUCCESS:
        return ExitStatus.SUCCESS
    if status is TaskStatus.PARTIAL:
        return ExitStatus.PARTIAL
    return ExitStatus.FAILED


async def run_orchestration(
    objective: str,
    *,
    config: Settings,
    state_store: StateStore | None = None,
    task_id: str = DEFAULT_ROOT_TASK_ID,
    run_id_factory: Callable[[], str] = create_run_id,
) -> RunnerResult:
    store = state_store or InMemoryStateStore()
    run_id = run_id_factory()
    deps = build_runtime_deps(config=config, state_store=store)
    await store.create_run(
        run_id,
        metadata={
            "objective": objective,
            "provider": deps.provider_label,
            "model": config.model_name,
        },
    )
    await store.set_task_status(run_id, task_id, TaskStatus.RUNNING)
    await store.append_trace_event(
        run_id,
        to_trace_event(RunStartedEvent(run_id=run_id, task_id=task_id, objective=objective)),
    )

    request = OrchestratorRequest(
        envelope=TaskEnvelope(run_id=run_id, task_id=task_id, objective=objective)
    )

    traced_resolver = _build_traced_capability_resolver(
        state_store=store,
        run_id=run_id,
        root_task_id=task_id,
    )

    terminal = await _execute_with_terminal_handling(
        deps=deps,
        request=request,
        timeout_seconds=config.timeout_seconds,
        traced_capability_resolver=traced_resolver,
        retry_notifier=_build_retry_notifier(
            state_store=store,
            run_id=run_id,
            root_task_id=task_id,
        ),
    )

    normalized_status = normalize_terminal_status(terminal.status)
    normalized_response = terminal.model_copy(update={"status": normalized_status})
    await _persist_terminal_state(
        state_store=store,
        run_id=run_id,
        task_id=task_id,
        response=normalized_response,
    )

    exit_status = _to_exit_status(normalized_status)
    return RunnerResult(
        **normalized_response.model_dump(),
        exit_status=exit_status,
        exit_code=0 if exit_status is ExitStatus.SUCCESS else 1,
    )


async def _execute_with_terminal_handling(
    *,
    deps: RuntimeDeps,
    request: OrchestratorRequest,
    timeout_seconds: float,
    traced_capability_resolver: Callable[[str], WorkerCapability[Any, Any]],
    retry_notifier: Callable[[str, int, Failure], asyncio.Future[None] | Any],
) -> FinalResponse:
    orchestration_timeout = timeout_seconds * max(1, deps.config.max_steps)
    try:
        return await asyncio.wait_for(
            execute_orchestrator(
                deps=deps,
                request=request,
                capability_resolver=traced_capability_resolver,
                on_retry=retry_notifier,
            ),
            timeout=orchestration_timeout,
        )
    except TimeoutError:
        timeout_failure = Failure(
            kind=FailureKind.TIMEOUT,
            message=("Runtime runner timed out before orchestration completed."),
            details={
                "worker_timeout_seconds": str(timeout_seconds),
                "orchestration_timeout_seconds": str(orchestration_timeout),
            },
        )
        return FinalResponse(
            run_id=request.envelope.run_id,
            status=TaskStatus.FAILED,
            summary="Runtime orchestration timed out.",
            failures=[timeout_failure],
        )
    except Exception as exc:
        runtime_failure = Failure(
            kind=_failure_kind_for_exception(exc),
            message="Runtime runner failed while executing orchestration.",
            details={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        return FinalResponse(
            run_id=request.envelope.run_id,
            status=TaskStatus.FAILED,
            summary="Runtime orchestration failed due to unexpected exception.",
            failures=[runtime_failure],
        )


async def _persist_terminal_state(
    *,
    state_store: StateStore,
    run_id: str,
    task_id: str,
    response: FinalResponse,
) -> None:
    for artifact in response.artifacts:
        await state_store.add_artifact(run_id, artifact)
    for failure in response.failures:
        await state_store.add_failure(run_id, failure)

    await state_store.set_task_status(run_id, task_id, response.status)

    if response.status is TaskStatus.SUCCESS:
        await state_store.append_trace_event(
            run_id,
            to_trace_event(
                RunCompletedEvent(
                    run_id=run_id,
                    task_id=task_id,
                    status=response.status,
                    summary=response.summary,
                )
            ),
        )
        return

    failure = (
        response.failures[-1]
        if response.failures
        else Failure(
            kind=FailureKind.WORKER,
            message=response.summary,
        )
    )
    await state_store.append_trace_event(
        run_id,
        to_trace_event(
            RunFailedEvent(
                run_id=run_id,
                task_id=task_id,
                status=response.status,
                failure_kind=failure.kind,
                message=failure.message,
            )
        ),
    )


def _build_traced_capability_resolver(
    *,
    state_store: StateStore,
    run_id: str,
    root_task_id: str,
) -> Callable[[str], WorkerCapability[Any, Any]]:
    attempts_by_worker: dict[str, int] = {}

    def traced_capability_resolver(worker_name: str) -> WorkerCapability[Any, Any]:
        original_capability = get_worker_capability(worker_name)

        async def execute_with_tracing(deps: RuntimeDeps, request: Any) -> Any:
            attempt = attempts_by_worker.get(worker_name, 0) + 1
            attempts_by_worker[worker_name] = attempt
            worker_task_id = f"{root_task_id}:{worker_name}"
            started = True

            try:
                await state_store.append_trace_event(
                    run_id,
                    to_trace_event(
                        WorkerDispatchedEvent(
                            run_id=run_id,
                            task_id=worker_task_id,
                            worker_name=worker_name,
                            attempt=attempt,
                        )
                    ),
                )
                await state_store.set_task_status(run_id, worker_task_id, TaskStatus.RUNNING)
                response = await asyncio.wait_for(
                    original_capability.execute(deps, request),
                    timeout=deps.config.timeout_seconds,
                )
            except asyncio.CancelledError:
                if started:
                    await _persist_worker_completion(
                        state_store=state_store,
                        run_id=run_id,
                        worker_task_id=worker_task_id,
                        worker_name=worker_name,
                        attempt=attempt,
                        status=TaskStatus.FAILED,
                        failure_kind=FailureKind.TIMEOUT,
                    )
                raise
            except TimeoutError:
                await _persist_worker_completion(
                    state_store=state_store,
                    run_id=run_id,
                    worker_task_id=worker_task_id,
                    worker_name=worker_name,
                    attempt=attempt,
                    status=TaskStatus.FAILED,
                    failure_kind=FailureKind.TIMEOUT,
                )
                raise
            except Exception as exc:
                if started:
                    await _persist_worker_completion(
                        state_store=state_store,
                        run_id=run_id,
                        worker_task_id=worker_task_id,
                        worker_name=worker_name,
                        attempt=attempt,
                        status=TaskStatus.FAILED,
                        failure_kind=_failure_kind_for_exception(exc),
                    )
                raise

            response_status = normalize_terminal_status(
                getattr(response, "status", TaskStatus.FAILED)
            )
            failure = getattr(response, "failure", None)
            await _persist_worker_completion(
                state_store=state_store,
                run_id=run_id,
                worker_task_id=worker_task_id,
                worker_name=worker_name,
                attempt=attempt,
                status=response_status,
                failure_kind=failure.kind if failure is not None else None,
            )

            return response

        return WorkerCapability(
            name=original_capability.name,
            request_model=original_capability.request_model,
            response_model=original_capability.response_model,
            execute=execute_with_tracing,
            request_factory=original_capability.request_factory,
            fallback_retryable=original_capability.fallback_retryable,
            success_summary=original_capability.success_summary,
            failure_terminal_status=original_capability.failure_terminal_status,
            max_steps_summary=original_capability.max_steps_summary,
            non_retryable_summary=original_capability.non_retryable_summary,
            retry_exhausted_summary=original_capability.retry_exhausted_summary,
        )

    return traced_capability_resolver


async def _persist_worker_completion(
    *,
    state_store: StateStore,
    run_id: str,
    worker_task_id: str,
    worker_name: str,
    attempt: int,
    status: TaskStatus,
    failure_kind: FailureKind | None,
) -> None:
    await state_store.set_task_status(run_id, worker_task_id, status)
    await state_store.append_trace_event(
        run_id,
        to_trace_event(
            WorkerCompletedEvent(
                run_id=run_id,
                task_id=worker_task_id,
                worker_name=worker_name,
                attempt=attempt,
                status=status,
                failure_kind=failure_kind,
            )
        ),
    )


async def _emit_retry_trace(
    *,
    state_store: StateStore,
    run_id: str,
    root_task_id: str,
    worker_name: str,
    attempt: int,
    failure: Failure,
) -> None:
    await state_store.append_trace_event(
        run_id,
        to_trace_event(
            WorkerRetryEvent(
                run_id=run_id,
                task_id=f"{root_task_id}:{worker_name}",
                worker_name=worker_name,
                attempt=attempt,
                reason=failure.message,
                failure_kind=failure.kind,
            )
        ),
    )


def _build_retry_notifier(
    *,
    state_store: StateStore,
    run_id: str,
    root_task_id: str,
) -> Callable[[str, int, Failure], Any]:
    async def notify(worker_name: str, attempt: int, failure: Failure) -> None:
        await _emit_retry_trace(
            state_store=state_store,
            run_id=run_id,
            root_task_id=root_task_id,
            worker_name=worker_name,
            attempt=attempt,
            failure=failure,
        )

    return notify


def _failure_kind_for_exception(exc: Exception) -> FailureKind:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return FailureKind.TIMEOUT
    if isinstance(exc, UnknownWorkerCapabilityError):
        return FailureKind.CONFIG
    if isinstance(exc, (ValidationError, ValueError)):
        return FailureKind.VALIDATION
    return FailureKind.WORKER
