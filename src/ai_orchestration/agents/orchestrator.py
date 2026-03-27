from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from ai_orchestration.agents.base import WorkerCapability
from ai_orchestration.agents.registry import get_worker_capability
from ai_orchestration.contracts.coder import CoderArtifact, CoderRequest
from ai_orchestration.contracts.common import Artifact, Failure, FailureKind, TaskStatus
from ai_orchestration.contracts.orchestrator import FinalResponse, OrchestratorRequest
from ai_orchestration.contracts.researcher import ResearcherArtifact, ResearcherRequest
from ai_orchestration.contracts.reviewer import ReviewerRequest
from ai_orchestration.deps import RuntimeDeps

CapabilityResolver = Callable[[str], WorkerCapability[Any, Any]]
RetryNotifier = Callable[[str, int, Failure], Awaitable[None]]


def _to_common_artifacts(
    worker_artifacts: Sequence[BaseModel | Mapping[str, object]],
) -> list[Artifact]:
    common_artifacts: list[Artifact] = []
    for artifact in worker_artifacts:
        payload = artifact.model_dump() if isinstance(artifact, BaseModel) else dict(artifact)
        common_artifacts.append(Artifact.model_validate(payload))
    return common_artifacts


def _fallback_failure(worker_name: str, status: TaskStatus, retryable: bool) -> Failure:
    return Failure(
        kind=FailureKind.WORKER,
        message=f"{worker_name} returned {status.value} without failure details.",
        retryable=retryable,
        details={"worker": worker_name},
    )


def _max_steps_failure(worker_name: str, max_steps: int) -> Failure:
    return Failure(
        kind=FailureKind.ORCHESTRATION_LIMIT,
        message="Orchestrator reached max_steps before completing worker sequence.",
        details={"worker": worker_name, "max_steps": str(max_steps)},
    )


def _failed_response(
    run_id: str, summary: str, artifacts: list[Artifact], failures: list[Failure]
) -> FinalResponse:
    return FinalResponse(
        run_id=run_id,
        status=TaskStatus.FAILED,
        summary=summary,
        artifacts=artifacts,
        failures=failures,
    )


def _review_terminal_status(candidate_artifacts: list[CoderArtifact]) -> TaskStatus:
    return TaskStatus.PARTIAL if candidate_artifacts else TaskStatus.FAILED


async def execute_orchestrator(
    deps: RuntimeDeps,
    request: OrchestratorRequest,
    *,
    capability_resolver: CapabilityResolver | None = None,
    on_retry: RetryNotifier | None = None,
) -> FinalResponse:
    resolve_capability = capability_resolver or get_worker_capability
    envelope = request.envelope
    retry_budget = deps.config.retry_budget
    max_steps = deps.config.max_steps
    steps_used = 0

    artifacts: list[Artifact] = list(envelope.input_artifacts)
    failures: list[Failure] = []

    researcher_capability = resolve_capability("researcher")
    researcher_request = ResearcherRequest(
        run_id=envelope.run_id,
        task_id=envelope.task_id,
        objective=envelope.objective,
        constraints=envelope.constraints,
    )
    researcher_artifacts: list[ResearcherArtifact] = []

    for attempt in range(retry_budget + 1):
        if steps_used >= max_steps:
            limit_failure = _max_steps_failure("researcher", max_steps)
            return _failed_response(
                run_id=envelope.run_id,
                summary="Orchestrator exceeded max_steps before researcher completed.",
                artifacts=artifacts,
                failures=[*failures, limit_failure],
            )

        steps_used += 1
        researcher_response = await researcher_capability.execute(deps, researcher_request)
        if researcher_response.status is TaskStatus.SUCCESS:
            researcher_artifacts = list(researcher_response.artifacts)
            artifacts.extend(_to_common_artifacts(researcher_response.artifacts))
            break

        failure = researcher_response.failure or _fallback_failure(
            worker_name="researcher",
            status=researcher_response.status,
            retryable=False,
        )
        failures.append(failure)

        if not failure.retryable:
            return _failed_response(
                run_id=envelope.run_id,
                summary="Researcher failed with non-retryable error.",
                artifacts=artifacts,
                failures=failures,
            )
        if attempt == retry_budget:
            return _failed_response(
                run_id=envelope.run_id,
                summary="Researcher retry budget exhausted.",
                artifacts=artifacts,
                failures=failures,
            )
        if on_retry is not None:
            await on_retry("researcher", attempt + 1, failure)

    coder_capability = resolve_capability("coder")
    coder_request = CoderRequest(
        run_id=envelope.run_id,
        task_id=envelope.task_id,
        objective=envelope.objective,
        research_artifacts=researcher_artifacts,
        constraints=envelope.constraints,
    )
    coder_artifacts: list[CoderArtifact] = []

    for attempt in range(retry_budget + 1):
        if steps_used >= max_steps:
            limit_failure = _max_steps_failure("coder", max_steps)
            return _failed_response(
                run_id=envelope.run_id,
                summary="Orchestrator exceeded max_steps before coder completed.",
                artifacts=artifacts,
                failures=[*failures, limit_failure],
            )

        steps_used += 1
        coder_response = await coder_capability.execute(deps, coder_request)
        if coder_response.status is TaskStatus.SUCCESS:
            coder_artifacts = list(coder_response.artifacts)
            artifacts.extend(_to_common_artifacts(coder_response.artifacts))
            break

        failure = coder_response.failure or _fallback_failure(
            worker_name="coder",
            status=coder_response.status,
            retryable=False,
        )
        failures.append(failure)

        if not failure.retryable:
            return _failed_response(
                run_id=envelope.run_id,
                summary="Coder failed with non-retryable error.",
                artifacts=artifacts,
                failures=failures,
            )
        if attempt == retry_budget:
            return _failed_response(
                run_id=envelope.run_id,
                summary="Coder retry budget exhausted.",
                artifacts=artifacts,
                failures=failures,
            )
        if on_retry is not None:
            await on_retry("coder", attempt + 1, failure)

    reviewer_capability = resolve_capability("reviewer")
    reviewer_request = ReviewerRequest(
        run_id=envelope.run_id,
        task_id=envelope.task_id,
        objective=envelope.objective,
        candidate_artifacts=coder_artifacts,
        constraints=envelope.constraints,
    )

    for attempt in range(retry_budget + 1):
        if steps_used >= max_steps:
            limit_failure = _max_steps_failure("reviewer", max_steps)
            terminal_status = _review_terminal_status(coder_artifacts)
            return FinalResponse(
                run_id=envelope.run_id,
                status=terminal_status,
                summary="Orchestrator exceeded max_steps before reviewer completed.",
                artifacts=artifacts,
                failures=[*failures, limit_failure],
            )

        steps_used += 1
        reviewer_response = await reviewer_capability.execute(deps, reviewer_request)
        if reviewer_response.status is TaskStatus.SUCCESS:
            artifacts.extend(_to_common_artifacts(reviewer_response.artifacts))
            return FinalResponse(
                run_id=envelope.run_id,
                status=TaskStatus.SUCCESS,
                summary="Completed researcher -> coder -> reviewer sequence successfully.",
                artifacts=artifacts,
                failures=failures,
            )

        failure = reviewer_response.failure or _fallback_failure(
            worker_name="reviewer",
            status=reviewer_response.status,
            retryable=True,
        )
        failures.append(failure)

        if not failure.retryable:
            terminal_status = _review_terminal_status(coder_artifacts)
            return FinalResponse(
                run_id=envelope.run_id,
                status=terminal_status,
                summary="Reviewer rejected candidate output with non-retryable failure.",
                artifacts=artifacts,
                failures=failures,
            )
        if attempt == retry_budget:
            terminal_status = _review_terminal_status(coder_artifacts)
            return FinalResponse(
                run_id=envelope.run_id,
                status=terminal_status,
                summary="Reviewer rejected candidate output after retry budget exhausted.",
                artifacts=artifacts,
                failures=failures,
            )
        if on_retry is not None:
            await on_retry("reviewer", attempt + 1, failure)

    return _failed_response(
        run_id=envelope.run_id,
        summary="Unexpected orchestrator termination.",
        artifacts=artifacts,
        failures=failures,
    )
