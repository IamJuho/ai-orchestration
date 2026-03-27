import asyncio
from collections.abc import Mapping

from ai_orchestration.contracts.common import Artifact, Failure, TaskStatus

from .base import RunState, TraceEvent


class InMemoryStateStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}
        self._lock = asyncio.Lock()

    async def create_run(
        self,
        run_id: str,
        metadata: Mapping[str, str] | None = None,
    ) -> RunState:
        async with self._lock:
            existing = self._runs.get(run_id)
            if existing is not None:
                return existing.model_copy(deep=True)

            created = RunState(run_id=run_id, metadata=dict(metadata or {}))
            self._runs[run_id] = created
            return created.model_copy(deep=True)

    async def get_run(self, run_id: str) -> RunState | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return run.model_copy(deep=True)

    async def set_task_status(self, run_id: str, task_id: str, status: TaskStatus) -> RunState:
        async with self._lock:
            run = self._require_run(run_id)
            updated_statuses = {**run.task_statuses, task_id: status}
            updated = run.model_copy(update={"task_statuses": updated_statuses})
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    async def add_artifact(self, run_id: str, artifact: Artifact) -> RunState:
        async with self._lock:
            run = self._require_run(run_id)
            updated = run.model_copy(update={"artifacts": (*run.artifacts, artifact)})
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    async def add_failure(self, run_id: str, failure: Failure) -> RunState:
        async with self._lock:
            run = self._require_run(run_id)
            updated = run.model_copy(update={"failures": (*run.failures, failure)})
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    async def append_trace_event(self, run_id: str, event: TraceEvent) -> RunState:
        async with self._lock:
            run = self._require_run(run_id)
            updated = run.model_copy(update={"trace_events": (*run.trace_events, event)})
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    def _require_run(self, run_id: str) -> RunState:
        run = self._runs.get(run_id)
        if run is None:
            msg = f"Run state '{run_id}' was not created"
            raise KeyError(msg)
        return run
