import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from ai_orchestration.contracts.coder import CoderRequest, CoderResponse
from ai_orchestration.contracts.common import FailureKind, TaskStatus
from ai_orchestration.contracts.researcher import ResearcherRequest, ResearcherResponse
from ai_orchestration.contracts.reviewer import ReviewerRequest, ReviewerResponse
from ai_orchestration.deps import RuntimeDeps
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.runner import run_orchestration
from ai_orchestration.state.memory import InMemoryStateStore


class StubCapability:
    def __init__(self, execute: Callable[[RuntimeDeps, Any], Awaitable[Any]]) -> None:
        self.execute = execute
        self.name = "stub"
        self.request_model = object
        self.response_model = object


def _settings(*, timeout_seconds: float) -> Settings:
    return Settings.model_validate(
        {
            "OPENAI_API_KEY": "test-key",
            "timeout_seconds": timeout_seconds,
            "retry_budget": 1,
        }
    )


@pytest.mark.asyncio
async def test_runner_timeout_persists_terminal_failure_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def execute_researcher(_: RuntimeDeps, __: ResearcherRequest) -> ResearcherResponse:
        await asyncio.sleep(0.05)
        return ResearcherResponse(status=TaskStatus.SUCCESS)

    async def execute_coder(_: RuntimeDeps, __: CoderRequest) -> CoderResponse:
        raise AssertionError("Coder should not execute after researcher timeout")

    async def execute_reviewer(_: RuntimeDeps, __: ReviewerRequest) -> ReviewerResponse:
        raise AssertionError("Reviewer should not execute after researcher timeout")

    capabilities: dict[str, StubCapability] = {
        "researcher": StubCapability(execute=execute_researcher),
        "coder": StubCapability(execute=execute_coder),
        "reviewer": StubCapability(execute=execute_reviewer),
    }

    monkeypatch.setattr(
        "ai_orchestration.runtime.runner.get_worker_capability",
        lambda name: capabilities[name],
    )

    store = InMemoryStateStore()
    result = await run_orchestration(
        "Ship timeout handling",
        config=_settings(timeout_seconds=0.01),
        state_store=store,
        task_id="task-timeout",
        run_id_factory=lambda: "run-timeout",
    )

    assert result.status is TaskStatus.FAILED
    assert result.exit_status.value == "failed"
    assert result.exit_code == 1
    assert result.failures[-1].kind is FailureKind.TIMEOUT

    run_state = await store.get_run("run-timeout")
    assert run_state is not None
    assert run_state.task_statuses["task-timeout"] is TaskStatus.FAILED
    assert run_state.task_statuses["task-timeout:researcher"] is TaskStatus.FAILED
    assert tuple(event.event_type for event in run_state.trace_events) == (
        "run.started",
        "worker.dispatched",
        "worker.completed",
        "run.failed",
    )
    assert run_state.trace_events[2].payload["failure_kind"] == "timeout"
    assert run_state.trace_events[-1].payload["failure_kind"] == "timeout"
