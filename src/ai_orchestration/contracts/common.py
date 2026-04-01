from enum import StrEnum

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class FailureKind(StrEnum):
    CONFIG = "config"
    PROVIDER = "provider"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    ORCHESTRATION_LIMIT = "orchestration_limit"
    WORKER = "worker"


class Failure(BaseModel):
    kind: FailureKind
    message: str
    retryable: bool = False
    details: dict[str, str] = Field(default_factory=dict)


class Artifact(BaseModel):
    artifact_id: str
    kind: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class TaskEnvelope(BaseModel):
    run_id: str
    task_id: str
    objective: str
    input_artifacts: list[Artifact] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
