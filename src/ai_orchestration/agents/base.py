from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from ai_orchestration.contracts.common import Artifact, Failure, TaskStatus
from ai_orchestration.contracts.orchestrator import OrchestratorRequest
from ai_orchestration.deps import RuntimeDeps

WorkerRequestT = TypeVar("WorkerRequestT", bound=BaseModel)
WorkerResponseT = TypeVar("WorkerResponseT", bound=BaseModel)
WorkerRequestContraT = TypeVar("WorkerRequestContraT", bound=BaseModel, contravariant=True)
WorkerResponseCoT = TypeVar("WorkerResponseCoT", bound=BaseModel, covariant=True)


class WorkerExecutor(Protocol[WorkerRequestContraT, WorkerResponseCoT]):
    async def __call__(
        self, deps: RuntimeDeps, request: WorkerRequestContraT
    ) -> WorkerResponseCoT: ...


@dataclass
class OrchestrationContext:
    artifacts: list[Artifact] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    artifacts_by_worker: dict[str, list[Artifact]] = field(default_factory=dict)


def default_failure_terminal_status(_: OrchestrationContext) -> TaskStatus:
    return TaskStatus.FAILED


@dataclass(frozen=True)
class WorkerCapability(Generic[WorkerRequestT, WorkerResponseT]):
    name: str
    request_model: type[WorkerRequestT]
    response_model: type[WorkerResponseT]
    execute: WorkerExecutor[WorkerRequestT, WorkerResponseT]
    request_factory: Callable[[OrchestratorRequest, OrchestrationContext], WorkerRequestT]
    fallback_retryable: bool = False
    failure_terminal_status: Callable[[OrchestrationContext], TaskStatus] = (
        default_failure_terminal_status
    )


class UnknownWorkerCapabilityError(LookupError):
    def __init__(self, capability: str, available_capabilities: tuple[str, ...]) -> None:
        self.capability = capability
        self.available_capabilities = available_capabilities
        super().__init__(
            "Unknown worker capability "
            f"{capability!r}. Available capabilities: {', '.join(available_capabilities)}"
        )
