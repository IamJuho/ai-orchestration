from pydantic import ValidationError
from pydantic_ai import Agent, AgentRunError, UnexpectedModelBehavior

from ai_orchestration.contracts.common import Failure, FailureKind, TaskStatus
from ai_orchestration.contracts.researcher import ResearcherRequest, ResearcherResponse
from ai_orchestration.deps import RuntimeDeps

researcher_agent = Agent(
    deps_type=RuntimeDeps,
    output_type=ResearcherResponse,
    instructions="You are the Researcher. Return concise, factual research notes only.",
)


async def execute_researcher(deps: RuntimeDeps, request: ResearcherRequest) -> ResearcherResponse:
    try:
        result = await researcher_agent.run(
            user_prompt=(
                f"Objective: {request.objective}\n"
                f"Constraints: {', '.join(request.constraints) if request.constraints else 'none'}"
            ),
            deps=deps,
            model=deps.provider.model_name(),
        )
        return ResearcherResponse.model_validate(result.output)
    except (ValidationError, UnexpectedModelBehavior) as exc:
        return ResearcherResponse(
            status=TaskStatus.FAILED,
            failure=Failure(
                kind=FailureKind.VALIDATION,
                message="Researcher produced an invalid response payload.",
                details={"error": str(exc)},
            ),
        )
    except AgentRunError as exc:
        return ResearcherResponse(
            status=TaskStatus.FAILED,
            failure=Failure(
                kind=FailureKind.PROVIDER,
                message="Researcher provider execution failed.",
                retryable=True,
                details={"error": str(exc), "provider": deps.provider_label},
            ),
        )
