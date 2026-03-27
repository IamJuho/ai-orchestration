from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, Field

from ai_orchestration.contracts.common import Artifact, Failure, TaskStatus


class TraceEvent(BaseModel):
    event_type: str
    payload: dict[str, str] = Field(default_factory=dict)


class RunState(BaseModel):
    run_id: str
    metadata: dict[str, str] = Field(default_factory=dict)
    task_statuses: dict[str, TaskStatus] = Field(default_factory=dict)
    artifacts: tuple[Artifact, ...] = ()
    failures: tuple[Failure, ...] = ()
    trace_events: tuple[TraceEvent, ...] = ()


class StateStore(Protocol):
    async def create_run(
        self,
        run_id: str,
        metadata: Mapping[str, str] | None = None,
    ) -> RunState: ...

    async def get_run(self, run_id: str) -> RunState | None: ...

    async def set_task_status(self, run_id: str, task_id: str, status: TaskStatus) -> RunState: ...

    async def add_artifact(self, run_id: str, artifact: Artifact) -> RunState: ...

    async def add_failure(self, run_id: str, failure: Failure) -> RunState: ...

    async def append_trace_event(self, run_id: str, event: TraceEvent) -> RunState: ...
