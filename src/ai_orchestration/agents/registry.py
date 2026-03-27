from __future__ import annotations

from typing import Any, Literal, cast, overload

from ai_orchestration.agents.base import UnknownWorkerCapabilityError, WorkerCapability
from ai_orchestration.agents.coder import build_coder_request, execute_coder
from ai_orchestration.agents.researcher import build_researcher_request, execute_researcher
from ai_orchestration.agents.reviewer import (
    build_reviewer_request,
    execute_reviewer,
    reviewer_non_retryable_summary,
    reviewer_retry_exhausted_summary,
    reviewer_terminal_status,
)
from ai_orchestration.contracts.coder import CoderRequest, CoderResponse
from ai_orchestration.contracts.researcher import ResearcherRequest, ResearcherResponse
from ai_orchestration.contracts.reviewer import ReviewerRequest, ReviewerResponse

CapabilityName = Literal["researcher", "coder", "reviewer"]

_WORKER_CAPABILITIES: dict[CapabilityName, WorkerCapability[Any, Any]] = {
    "researcher": WorkerCapability(
        name="researcher",
        request_model=ResearcherRequest,
        response_model=ResearcherResponse,
        execute=execute_researcher,
        request_factory=build_researcher_request,
    ),
    "coder": WorkerCapability(
        name="coder",
        request_model=CoderRequest,
        response_model=CoderResponse,
        execute=execute_coder,
        request_factory=build_coder_request,
    ),
    "reviewer": WorkerCapability(
        name="reviewer",
        request_model=ReviewerRequest,
        response_model=ReviewerResponse,
        execute=execute_reviewer,
        request_factory=build_reviewer_request,
        fallback_retryable=True,
        terminal_on_success=True,
        success_summary="Completed researcher -> coder -> reviewer sequence successfully.",
        failure_terminal_status=reviewer_terminal_status,
        non_retryable_summary=reviewer_non_retryable_summary,
        retry_exhausted_summary=reviewer_retry_exhausted_summary,
    ),
}


def list_worker_capabilities() -> tuple[CapabilityName, ...]:
    return tuple(_WORKER_CAPABILITIES.keys())


@overload
def get_worker_capability(
    name: Literal["researcher"],
) -> WorkerCapability[ResearcherRequest, ResearcherResponse]: ...


@overload
def get_worker_capability(
    name: Literal["coder"],
) -> WorkerCapability[CoderRequest, CoderResponse]: ...


@overload
def get_worker_capability(
    name: Literal["reviewer"],
) -> WorkerCapability[ReviewerRequest, ReviewerResponse]: ...


@overload
def get_worker_capability(name: str) -> WorkerCapability[Any, Any]: ...


def get_worker_capability(name: str) -> WorkerCapability[Any, Any]:
    if name in _WORKER_CAPABILITIES:
        return _WORKER_CAPABILITIES[cast(CapabilityName, name)]

    raise UnknownWorkerCapabilityError(
        capability=name,
        available_capabilities=list_worker_capabilities(),
    )
