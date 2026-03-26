from typing import Literal, Optional

from pydantic import BaseModel, Field

from .common import Failure, TaskStatus
from .researcher import ResearcherArtifact


class CoderRequest(BaseModel):
    run_id: str
    task_id: str
    objective: str
    research_artifacts: list[ResearcherArtifact] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class CoderArtifact(BaseModel):
    artifact_id: str
    kind: Literal["implementation_plan"] = "implementation_plan"
    producer: Literal["coder"] = "coder"
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class CoderResponse(BaseModel):
    status: TaskStatus
    artifacts: list[CoderArtifact] = Field(default_factory=list)
    failure: Optional[Failure] = None
