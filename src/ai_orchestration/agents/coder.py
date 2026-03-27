from pydantic import ValidationError
from pydantic_ai import Agent, AgentRunError, UnexpectedModelBehavior

from ai_orchestration.contracts.coder import CoderRequest, CoderResponse
from ai_orchestration.contracts.common import Failure, FailureKind, TaskStatus
from ai_orchestration.deps import RuntimeDeps

coder_agent = Agent(
    deps_type=RuntimeDeps,
    output_type=CoderResponse,
    instructions="You are the Coder. Return an implementation plan artifact only.",
)


async def execute_coder(deps: RuntimeDeps, request: CoderRequest) -> CoderResponse:
    try:
        result = await coder_agent.run(
            user_prompt=(
                f"Objective: {request.objective}\n"
                f"Research artifacts: {len(request.research_artifacts)}\n"
                f"Constraints: {', '.join(request.constraints) if request.constraints else 'none'}"
            ),
            deps=deps,
            model=deps.provider.model_name(),
        )
        return CoderResponse.model_validate(result.output)
    except (ValidationError, UnexpectedModelBehavior) as exc:
        return CoderResponse(
            status=TaskStatus.FAILED,
            failure=Failure(
                kind=FailureKind.VALIDATION,
                message="Coder produced an invalid response payload.",
                details={"error": str(exc)},
            ),
        )
    except AgentRunError as exc:
        return CoderResponse(
            status=TaskStatus.FAILED,
            failure=Failure(
                kind=FailureKind.PROVIDER,
                message="Coder provider execution failed.",
                retryable=True,
                details={"error": str(exc), "provider": deps.provider_label},
            ),
        )
