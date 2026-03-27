import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai import AgentRunError

from ai_orchestration.agents import coder, researcher, reviewer
from ai_orchestration.contracts.common import FailureKind, TaskStatus
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.runner import run_orchestration
from ai_orchestration.state.memory import InMemoryStateStore


@dataclass
class StubRunResult:
    output: Any


class StubAgent:
    def __init__(self, worker_name: str, call_log: list[str], outputs: list[Any]) -> None:
        self._worker_name = worker_name
        self._call_log = call_log
        self._outputs = outputs
        self._index = 0

    async def run(self, **_: Any) -> StubRunResult:
        self._call_log.append(self._worker_name)
        if self._index >= len(self._outputs):
            msg = f"No scripted output available for {self._worker_name}"
            raise AssertionError(msg)

        current = self._outputs[self._index]
        self._index += 1
        if isinstance(current, Exception):
            raise current
        if callable(current):
            value = current()
            if asyncio.iscoroutine(value):
                value = await value
            return StubRunResult(output=value)
        return StubRunResult(output=current)


def _settings(*, timeout_seconds: float = 0.5, retry_budget: int = 1) -> Settings:
    return Settings.model_validate(
        {
            "OPENAI_API_KEY": "test-key",
            "timeout_seconds": timeout_seconds,
            "retry_budget": retry_budget,
        }
    )


@pytest.mark.asyncio
async def test_runner_integration_routes_workers_in_order_with_fake_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        researcher,
        "researcher_agent",
        StubAgent(
            worker_name="researcher",
            call_log=calls,
            outputs=[
                {
                    "status": TaskStatus.SUCCESS,
                    "artifacts": [{"artifact_id": "r-1", "content": "notes"}],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        coder,
        "coder_agent",
        StubAgent(
            worker_name="coder",
            call_log=calls,
            outputs=[
                {
                    "status": TaskStatus.SUCCESS,
                    "artifacts": [{"artifact_id": "c-1", "content": "plan"}],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        reviewer,
        "reviewer_agent",
        StubAgent(
            worker_name="reviewer",
            call_log=calls,
            outputs=[
                {
                    "status": TaskStatus.SUCCESS,
                    "artifacts": [{"artifact_id": "v-1", "content": "approved"}],
                }
            ],
        ),
    )

    store = InMemoryStateStore()
    result = await run_orchestration(
        "Route work across worker wrappers",
        config=_settings(retry_budget=1),
        state_store=store,
        task_id="task-routing",
        run_id_factory=lambda: "run-routing",
    )

    assert calls == ["researcher", "coder", "reviewer"]
    assert result.status is TaskStatus.SUCCESS
    assert [artifact.producer for artifact in result.artifacts] == [
        "researcher",
        "coder",
        "reviewer",
    ]

    run_state = await store.get_run("run-routing")
    assert run_state is not None
    assert tuple(event.event_type for event in run_state.trace_events) == (
        "run.started",
        "worker.dispatched",
        "worker.completed",
        "worker.dispatched",
        "worker.completed",
        "worker.dispatched",
        "worker.completed",
        "run.completed",
    )


@pytest.mark.asyncio
async def test_runner_integration_retries_retryable_failure_until_coder_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        researcher,
        "researcher_agent",
        StubAgent(
            worker_name="researcher",
            call_log=calls,
            outputs=[
                {
                    "status": TaskStatus.SUCCESS,
                    "artifacts": [{"artifact_id": "r-1", "content": "notes"}],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        coder,
        "coder_agent",
        StubAgent(
            worker_name="coder",
            call_log=calls,
            outputs=[
                AgentRunError("provider unavailable #1"),
                AgentRunError("provider unavailable #2"),
                AgentRunError("provider unavailable #3"),
            ],
        ),
    )
    monkeypatch.setattr(
        reviewer,
        "reviewer_agent",
        StubAgent(worker_name="reviewer", call_log=calls, outputs=[]),
    )

    store = InMemoryStateStore()
    result = await run_orchestration(
        "Exhaust coder retries",
        config=_settings(retry_budget=2),
        state_store=store,
        task_id="task-retries",
        run_id_factory=lambda: "run-retries",
    )

    assert calls == ["researcher", "coder", "coder", "coder"]
    assert result.status is TaskStatus.FAILED
    assert result.failures[-1].kind is FailureKind.PROVIDER

    run_state = await store.get_run("run-retries")
    assert run_state is not None
    retry_events = [event for event in run_state.trace_events if event.event_type == "worker.retry"]
    assert len(retry_events) == 2
    assert [event.payload["attempt"] for event in retry_events] == ["1", "2"]
    assert [event.payload["worker_name"] for event in retry_events] == ["coder", "coder"]


@pytest.mark.asyncio
async def test_runner_integration_fails_on_malformed_coder_output_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        researcher,
        "researcher_agent",
        StubAgent(
            worker_name="researcher",
            call_log=calls,
            outputs=[
                {
                    "status": TaskStatus.SUCCESS,
                    "artifacts": [{"artifact_id": "r-1", "content": "notes"}],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        coder,
        "coder_agent",
        StubAgent(
            worker_name="coder",
            call_log=calls,
            outputs=[
                {
                    "status": TaskStatus.SUCCESS,
                    "artifacts": [
                        {
                            "artifact_id": "bad-coder",
                            "kind": "review_report",
                            "producer": "reviewer",
                            "content": "wrong artifact",
                        }
                    ],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        reviewer,
        "reviewer_agent",
        StubAgent(worker_name="reviewer", call_log=calls, outputs=[]),
    )

    store = InMemoryStateStore()
    result = await run_orchestration(
        "Fail malformed coder payload",
        config=_settings(retry_budget=3),
        state_store=store,
        task_id="task-malformed",
        run_id_factory=lambda: "run-malformed",
    )

    assert calls == ["researcher", "coder"]
    assert result.status is TaskStatus.FAILED
    assert result.failures[-1].kind is FailureKind.VALIDATION

    run_state = await store.get_run("run-malformed")
    assert run_state is not None
    retry_events = [event for event in run_state.trace_events if event.event_type == "worker.retry"]
    assert retry_events == []


@pytest.mark.asyncio
async def test_runner_integration_reviewer_rejection_returns_partial_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        researcher,
        "researcher_agent",
        StubAgent(
            worker_name="researcher",
            call_log=calls,
            outputs=[
                {
                    "status": TaskStatus.SUCCESS,
                    "artifacts": [{"artifact_id": "r-1", "content": "notes"}],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        coder,
        "coder_agent",
        StubAgent(
            worker_name="coder",
            call_log=calls,
            outputs=[
                {
                    "status": TaskStatus.SUCCESS,
                    "artifacts": [{"artifact_id": "c-1", "content": "plan"}],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        reviewer,
        "reviewer_agent",
        StubAgent(
            worker_name="reviewer",
            call_log=calls,
            outputs=[
                {"status": TaskStatus.PARTIAL},
                {"status": TaskStatus.PARTIAL},
            ],
        ),
    )

    store = InMemoryStateStore()
    result = await run_orchestration(
        "Return partial after reviewer rejection",
        config=_settings(retry_budget=1),
        state_store=store,
        task_id="task-review",
        run_id_factory=lambda: "run-review",
    )

    assert calls == ["researcher", "coder", "reviewer", "reviewer"]
    assert result.status is TaskStatus.PARTIAL
    assert result.failures[-1].kind is FailureKind.WORKER

    run_state = await store.get_run("run-review")
    assert run_state is not None
    retry_events = [event for event in run_state.trace_events if event.event_type == "worker.retry"]
    assert len(retry_events) == 1
    assert retry_events[0].payload["worker_name"] == "reviewer"
    assert retry_events[0].payload["attempt"] == "1"
    reviewer_completed_events = [
        event
        for event in run_state.trace_events
        if event.event_type == "worker.completed" and event.payload["worker_name"] == "reviewer"
    ]
    assert [event.payload["attempt"] for event in reviewer_completed_events] == ["1", "2"]
    assert [event.payload["status"] for event in reviewer_completed_events] == [
        "partial",
        "partial",
    ]
    assert run_state.trace_events[-1].event_type == "run.failed"
    assert run_state.trace_events[-1].payload["status"] == "partial"


@pytest.mark.asyncio
async def test_runner_integration_times_out_before_slow_researcher_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def delayed_success() -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return {"status": TaskStatus.SUCCESS}

    monkeypatch.setattr(
        researcher,
        "researcher_agent",
        StubAgent(worker_name="researcher", call_log=calls, outputs=[delayed_success]),
    )
    monkeypatch.setattr(
        coder,
        "coder_agent",
        StubAgent(worker_name="coder", call_log=calls, outputs=[]),
    )
    monkeypatch.setattr(
        reviewer,
        "reviewer_agent",
        StubAgent(worker_name="reviewer", call_log=calls, outputs=[]),
    )

    store = InMemoryStateStore()
    result = await run_orchestration(
        "Timeout integration path",
        config=_settings(timeout_seconds=0.01, retry_budget=1),
        state_store=store,
        task_id="task-timeout-int",
        run_id_factory=lambda: "run-timeout-int",
    )

    assert calls == ["researcher"]
    assert result.status is TaskStatus.FAILED
    assert result.failures[-1].kind is FailureKind.TIMEOUT

    run_state = await store.get_run("run-timeout-int")
    assert run_state is not None
    assert run_state.task_statuses["task-timeout-int:researcher"] is TaskStatus.RUNNING
    assert tuple(event.event_type for event in run_state.trace_events) == (
        "run.started",
        "worker.dispatched",
        "run.failed",
    )
