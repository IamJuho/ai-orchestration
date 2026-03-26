from typing import Literal, Optional

from pydantic import BaseModel, Field

from .common import Failure, TaskStatus


class ResearcherRequest(BaseModel):
    run_id: str
    task_id: str
    objective: str
    constraints: list[str] = Field(default_factory=list)


class ResearcherArtifact(BaseModel):
    artifact_id: str
    kind: Literal["research_notes"] = "research_notes"
    producer: Literal["researcher"] = "researcher"
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class ResearcherResponse(BaseModel):
    status: TaskStatus
    artifacts: list[ResearcherArtifact] = Field(default_factory=list)
    failure: Optional[Failure] = None
