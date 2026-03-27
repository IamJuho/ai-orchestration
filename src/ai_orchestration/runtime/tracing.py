from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel

from ai_orchestration.contracts.common import FailureKind, TaskStatus
from ai_orchestration.state.base import TraceEvent


class RuntimeTraceEvent(BaseModel):
    event_type: str


class RunStartedEvent(RuntimeTraceEvent):
    event_type: Literal["run.started"] = "run.started"
    run_id: str
    task_id: str
    objective: str


class WorkerDispatchedEvent(RuntimeTraceEvent):
    event_type: Literal["worker.dispatched"] = "worker.dispatched"
    run_id: str
    task_id: str
    worker_name: str
    attempt: int


class WorkerCompletedEvent(RuntimeTraceEvent):
    event_type: Literal["worker.completed"] = "worker.completed"
    run_id: str
    task_id: str
    worker_name: str
    attempt: int
    status: TaskStatus
    failure_kind: FailureKind | None = None


class WorkerRetryEvent(RuntimeTraceEvent):
    event_type: Literal["worker.retry"] = "worker.retry"
    run_id: str
    task_id: str
    worker_name: str
    attempt: int
    reason: str
    failure_kind: FailureKind | None = None


class RunCompletedEvent(RuntimeTraceEvent):
    event_type: Literal["run.completed"] = "run.completed"
    run_id: str
    task_id: str
    status: TaskStatus
    summary: str


class RunFailedEvent(RuntimeTraceEvent):
    event_type: Literal["run.failed"] = "run.failed"
    run_id: str
    task_id: str
    status: TaskStatus
    failure_kind: FailureKind
    message: str


TraceLifecycleEvent = (
    RunStartedEvent
    | WorkerDispatchedEvent
    | WorkerCompletedEvent
    | WorkerRetryEvent
    | RunCompletedEvent
    | RunFailedEvent
)


def to_trace_event(event: TraceLifecycleEvent) -> TraceEvent:
    payload: dict[str, str] = {}
    for key, value in event.model_dump().items():
        if key == "event_type":
            continue
        payload[key] = _stringify_trace_value(value)
    return TraceEvent(event_type=event.event_type, payload=payload)


def _stringify_trace_value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)
