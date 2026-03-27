from pydantic import ValidationError
from pydantic_ai import Agent, AgentRunError, UnexpectedModelBehavior

from ai_orchestration.contracts.common import Failure, FailureKind, TaskStatus
from ai_orchestration.contracts.reviewer import ReviewerRequest, ReviewerResponse
from ai_orchestration.deps import RuntimeDeps

reviewer_agent = Agent(
    deps_type=RuntimeDeps,
    output_type=ReviewerResponse,
    instructions="You are the Reviewer. Return a review report with pass/fail guidance.",
)


async def execute_reviewer(deps: RuntimeDeps, request: ReviewerRequest) -> ReviewerResponse:
    try:
        result = await reviewer_agent.run(
            user_prompt=(
                f"Objective: {request.objective}\n"
                f"Candidate artifacts: {len(request.candidate_artifacts)}\n"
                f"Constraints: {', '.join(request.constraints) if request.constraints else 'none'}"
            ),
            deps=deps,
            model=deps.provider.model_name(),
        )
        return ReviewerResponse.model_validate(result.output)
    except (ValidationError, UnexpectedModelBehavior) as exc:
        return ReviewerResponse(
            status=TaskStatus.FAILED,
            failure=Failure(
                kind=FailureKind.VALIDATION,
                message="Reviewer produced an invalid response payload.",
                details={"error": str(exc)},
            ),
        )
    except AgentRunError as exc:
        return ReviewerResponse(
            status=TaskStatus.FAILED,
            failure=Failure(
                kind=FailureKind.PROVIDER,
                message="Reviewer provider execution failed.",
                retryable=True,
                details={"error": str(exc), "provider": deps.provider_label},
            ),
        )
