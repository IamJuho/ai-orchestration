from ai_orchestration.agents.coder import execute_coder
from ai_orchestration.agents.registry import get_worker_capability, list_worker_capabilities
from ai_orchestration.agents.researcher import execute_researcher
from ai_orchestration.agents.reviewer import execute_reviewer
from ai_orchestration.contracts.coder import CoderRequest, CoderResponse
from ai_orchestration.contracts.researcher import ResearcherRequest, ResearcherResponse
from ai_orchestration.contracts.reviewer import ReviewerRequest, ReviewerResponse


def test_registry_lists_exactly_three_v1_capabilities() -> None:
    assert list_worker_capabilities() == ("researcher", "coder", "reviewer")


def test_registry_resolves_researcher_capability_definition() -> None:
    capability = get_worker_capability("researcher")

    assert capability.name == "researcher"
    assert capability.request_model is ResearcherRequest
    assert capability.response_model is ResearcherResponse
    assert capability.execute is execute_researcher


def test_registry_resolves_coder_capability_definition() -> None:
    capability = get_worker_capability("coder")

    assert capability.name == "coder"
    assert capability.request_model is CoderRequest
    assert capability.response_model is CoderResponse
    assert capability.execute is execute_coder


def test_registry_resolves_reviewer_capability_definition() -> None:
    capability = get_worker_capability("reviewer")

    assert capability.name == "reviewer"
    assert capability.request_model is ReviewerRequest
    assert capability.response_model is ReviewerResponse
    assert capability.execute is execute_reviewer
