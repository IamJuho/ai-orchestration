from ai_orchestration.contracts.coder import CoderArtifact, CoderRequest, CoderResponse
from ai_orchestration.contracts.common import (
    Artifact,
    Failure,
    FailureKind,
    TaskEnvelope,
    TaskStatus,
)
from ai_orchestration.contracts.orchestrator import FinalResponse, OrchestratorRequest
from ai_orchestration.contracts.researcher import (
    ResearcherArtifact,
    ResearcherRequest,
    ResearcherResponse,
)
from ai_orchestration.contracts.reviewer import ReviewerArtifact, ReviewerRequest, ReviewerResponse


def test_common_contracts_happy_path() -> None:
    failure = Failure(kind=FailureKind.PROVIDER, message="transient upstream issue", retryable=True)
    artifact = Artifact(
        artifact_id="a-1",
        kind="research_notes",
        producer="researcher",
        content="topic notes",
    )
    envelope = TaskEnvelope(
        run_id="run-1",
        task_id="task-1",
        objective="Research and draft",
        input_artifacts=[artifact],
        constraints=["no network"],
    )

    assert failure.kind.value == "provider"
    assert envelope.input_artifacts[0].kind == "research_notes"


def test_role_contracts_happy_path() -> None:
    researcher_request = ResearcherRequest(run_id="run-1", task_id="t-r", objective="Investigate")
    researcher_response = ResearcherResponse(
        status=TaskStatus.SUCCESS,
        artifacts=[
            ResearcherArtifact(
                artifact_id="r-1",
                content="research output",
            )
        ],
    )

    coder_request = CoderRequest(
        run_id="run-1",
        task_id="t-c",
        objective="Implement",
        research_artifacts=researcher_response.artifacts,
    )
    coder_response = CoderResponse(
        status=TaskStatus.SUCCESS,
        artifacts=[
            CoderArtifact(
                artifact_id="c-1",
                content="implementation plan",
            )
        ],
    )

    reviewer_request = ReviewerRequest(
        run_id="run-1",
        task_id="t-v",
        objective="Review",
        candidate_artifacts=coder_response.artifacts,
    )
    reviewer_response = ReviewerResponse(
        status=TaskStatus.SUCCESS,
        artifacts=[
            ReviewerArtifact(
                artifact_id="v-1",
                content="looks good",
            )
        ],
    )

    assert researcher_request.objective == "Investigate"
    assert coder_request.research_artifacts[0].kind == "research_notes"
    assert reviewer_request.candidate_artifacts[0].kind == "implementation_plan"
    assert reviewer_response.artifacts[0].producer == "reviewer"


def test_orchestrator_final_response_happy_path() -> None:
    request = OrchestratorRequest(
        envelope=TaskEnvelope(
            run_id="run-2",
            task_id="task-2",
            objective="Ship response",
        )
    )
    final_response = FinalResponse(
        run_id=request.envelope.run_id,
        status=TaskStatus.SUCCESS,
        summary="Completed all worker steps",
        artifacts=[
            Artifact(
                artifact_id="f-1",
                kind="final_answer",
                producer="orchestrator",
                content="final answer",
            )
        ],
    )

    assert final_response.status is TaskStatus.SUCCESS
    assert final_response.artifacts[0].producer == "orchestrator"
