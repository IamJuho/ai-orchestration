from typing import Literal, Optional

from pydantic import BaseModel, Field

from .coder import CoderArtifact
from .common import Failure, TaskStatus


class ReviewerRequest(BaseModel):
    run_id: str
    task_id: str
    objective: str
    candidate_artifacts: list[CoderArtifact] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class ReviewerArtifact(BaseModel):
    artifact_id: str
    kind: Literal["review_report"] = "review_report"
    producer: Literal["reviewer"] = "reviewer"
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class ReviewerResponse(BaseModel):
    status: TaskStatus
    artifacts: list[ReviewerArtifact] = Field(default_factory=list)
    failure: Optional[Failure] = None
