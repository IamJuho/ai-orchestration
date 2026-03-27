import pytest

from ai_orchestration.state.base import TraceEvent
from ai_orchestration.state.memory import InMemoryStateStore


@pytest.mark.asyncio
async def test_trace_events_append_without_overwrite() -> None:
    store = InMemoryStateStore()
    await store.create_run("run-trace")

    first_state = await store.append_trace_event(
        "run-trace",
        TraceEvent(event_type="run.started", payload={"step": "1"}),
    )
    second_state = await store.append_trace_event(
        "run-trace",
        TraceEvent(event_type="worker.completed", payload={"worker": "researcher"}),
    )

    assert len(first_state.trace_events) == 1
    assert first_state.trace_events[0].event_type == "run.started"

    assert len(second_state.trace_events) == 2
    assert second_state.trace_events[0].event_type == "run.started"
    assert second_state.trace_events[1].event_type == "worker.completed"

    persisted = await store.get_run("run-trace")
    assert persisted is not None
    assert tuple(event.event_type for event in persisted.trace_events) == (
        "run.started",
        "worker.completed",
    )


@pytest.mark.asyncio
async def test_trace_snapshot_does_not_change_after_future_appends() -> None:
    store = InMemoryStateStore()
    await store.create_run("run-immutable")

    early_snapshot = await store.append_trace_event(
        "run-immutable",
        TraceEvent(event_type="run.started"),
    )

    await store.append_trace_event(
        "run-immutable",
        TraceEvent(event_type="worker.dispatched", payload={"worker": "coder"}),
    )

    assert tuple(event.event_type for event in early_snapshot.trace_events) == ("run.started",)

    latest = await store.get_run("run-immutable")
    assert latest is not None
    assert tuple(event.event_type for event in latest.trace_events) == (
        "run.started",
        "worker.dispatched",
    )
