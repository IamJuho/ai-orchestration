import pytest
from pydantic import ValidationError

from ai_orchestration.contracts.coder import CoderResponse
from ai_orchestration.contracts.common import Artifact, Failure, TaskEnvelope, TaskStatus
from ai_orchestration.contracts.orchestrator import FinalResponse
from ai_orchestration.contracts.researcher import ResearcherResponse
from ai_orchestration.contracts.reviewer import ReviewerRequest


def test_failure_kind_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        Failure.model_validate({"kind": "unknown", "message": "nope"})


def test_task_status_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        ResearcherResponse.model_validate({"status": "done"})


def test_artifact_rejects_missing_required_content() -> None:
    with pytest.raises(ValidationError):
        Artifact.model_validate(
            {
                "artifact_id": "a-1",
                "kind": "research_notes",
                "producer": "researcher",
            }
        )


def test_task_envelope_rejects_missing_objective() -> None:
    with pytest.raises(ValidationError):
        TaskEnvelope.model_validate({"run_id": "run-1", "task_id": "task-1"})


def test_final_response_rejects_missing_summary() -> None:
    with pytest.raises(ValidationError):
        FinalResponse.model_validate({"run_id": "run-1", "status": TaskStatus.SUCCESS})


def test_researcher_response_rejects_invalid_artifact_kind() -> None:
    with pytest.raises(ValidationError):
        ResearcherResponse.model_validate(
            {
                "status": TaskStatus.SUCCESS,
                "artifacts": [
                    {
                        "artifact_id": "bad-1",
                        "kind": "implementation_plan",
                        "producer": "researcher",
                        "content": "wrong kind",
                    }
                ],
            }
        )


def test_coder_response_rejects_invalid_producer() -> None:
    with pytest.raises(ValidationError):
        CoderResponse.model_validate(
            {
                "status": TaskStatus.SUCCESS,
                "artifacts": [
                    {
                        "artifact_id": "bad-2",
                        "kind": "implementation_plan",
                        "producer": "researcher",
                        "content": "wrong producer",
                    }
                ],
            }
        )


def test_reviewer_request_rejects_invalid_candidate_payload() -> None:
    with pytest.raises(ValidationError):
        ReviewerRequest.model_validate(
            {
                "run_id": "run-1",
                "task_id": "task-1",
                "objective": "review",
                "candidate_artifacts": [
                    {
                        "artifact_id": "x",
                        "kind": "review_report",
                        "producer": "reviewer",
                        "content": "not a coder artifact",
                    }
                ],
            }
        )


def test_reviewer_request_rejects_candidate_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        ReviewerRequest.model_validate(
            {
                "run_id": "run-1",
                "task_id": "task-1",
                "objective": "review",
                "candidate_artifacts": [
                    {
                        "artifact_id": "x",
                        "kind": "implementation_plan",
                        "producer": "coder",
                    }
                ],
            }
        )
