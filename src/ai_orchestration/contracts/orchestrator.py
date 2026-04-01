from pydantic import BaseModel, Field

from .common import Artifact, Failure, TaskEnvelope, TaskStatus


class OrchestratorRequest(BaseModel):
    envelope: TaskEnvelope


class FinalResponse(BaseModel):
    run_id: str
    status: TaskStatus
    summary: str
    artifacts: list[Artifact] = Field(default_factory=list)
    failures: list[Failure] = Field(default_factory=list)
