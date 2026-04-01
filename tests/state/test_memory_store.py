import pytest

from ai_orchestration.contracts.common import Artifact, Failure, FailureKind, TaskStatus
from ai_orchestration.state.memory import InMemoryStateStore


@pytest.mark.asyncio
async def test_in_memory_store_create_get_and_update_run_state() -> None:
    store = InMemoryStateStore()

    created = await store.create_run("run-1", metadata={"objective": "ship task"})
    assert created.run_id == "run-1"
    assert created.metadata == {"objective": "ship task"}
    assert created.task_statuses == {}
    assert created.artifacts == ()
    assert created.failures == ()
    assert created.trace_events == ()

    await store.set_task_status("run-1", "research", TaskStatus.RUNNING)
    await store.set_task_status("run-1", "research", TaskStatus.SUCCESS)

    await store.add_artifact(
        "run-1",
        Artifact(
            artifact_id="a-1",
            kind="research_notes",
            producer="researcher",
            content="research output",
        ),
    )
    await store.add_failure(
        "run-1",
        Failure(kind=FailureKind.WORKER, message="reviewer rejected", retryable=True),
    )

    loaded = await store.get_run("run-1")
    assert loaded is not None
    assert loaded.task_statuses["research"] is TaskStatus.SUCCESS
    assert loaded.artifacts[0].artifact_id == "a-1"
    assert loaded.failures[0].kind is FailureKind.WORKER
    assert loaded.failures[0].retryable is True


@pytest.mark.asyncio
async def test_in_memory_store_get_returns_isolated_snapshot() -> None:
    store = InMemoryStateStore()
    await store.create_run("run-2")
    await store.set_task_status("run-2", "coder", TaskStatus.RUNNING)

    snapshot = await store.get_run("run-2")
    assert snapshot is not None
    snapshot.task_statuses["coder"] = TaskStatus.FAILED

    latest = await store.get_run("run-2")
    assert latest is not None
    assert latest.task_statuses["coder"] is TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_create_run_existing_returns_deep_isolated_snapshot() -> None:
    store = InMemoryStateStore()
    initial = await store.create_run("run-3", metadata={"objective": "initial"})
    initial.metadata["objective"] = "mutated"

    existing = await store.create_run("run-3")
    assert existing.metadata == {"objective": "initial"}


@pytest.mark.asyncio
async def test_add_artifact_snapshot_metadata_mutation_does_not_persist() -> None:
    store = InMemoryStateStore()
    await store.create_run("run-4")

    snapshot = await store.add_artifact(
        "run-4",
        Artifact(
            artifact_id="a-2",
            kind="research_notes",
            producer="researcher",
            content="notes",
            metadata={"source": "v1"},
        ),
    )
    snapshot.artifacts[0].metadata["source"] = "mutated"

    latest = await store.get_run("run-4")
    assert latest is not None
    assert latest.artifacts[0].metadata["source"] == "v1"
