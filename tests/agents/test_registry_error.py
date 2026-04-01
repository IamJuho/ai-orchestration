import pytest

from ai_orchestration.agents.base import UnknownWorkerCapabilityError
from ai_orchestration.agents.registry import get_worker_capability


def test_registry_unknown_capability_raises_typed_error() -> None:
    with pytest.raises(UnknownWorkerCapabilityError) as exc_info:
        get_worker_capability("planner")

    error = exc_info.value
    assert error.capability == "planner"
    assert error.available_capabilities == ("researcher", "coder", "reviewer")
    assert "Unknown worker capability 'planner'" in str(error)
