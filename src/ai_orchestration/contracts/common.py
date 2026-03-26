from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class FailureKind(str, Enum):
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
    kind: Literal["research_notes", "implementation_plan", "review_report", "final_answer"]
    producer: Literal["orchestrator", "researcher", "coder", "reviewer"]
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class TaskEnvelope(BaseModel):
    run_id: str
    task_id: str
    objective: str
    input_artifacts: list[Artifact] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
