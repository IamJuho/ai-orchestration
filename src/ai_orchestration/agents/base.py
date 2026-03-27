from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from ai_orchestration.deps import RuntimeDeps

WorkerRequestT = TypeVar("WorkerRequestT", bound=BaseModel)
WorkerResponseT = TypeVar("WorkerResponseT", bound=BaseModel)
WorkerRequestContraT = TypeVar("WorkerRequestContraT", bound=BaseModel, contravariant=True)
WorkerResponseCoT = TypeVar("WorkerResponseCoT", bound=BaseModel, covariant=True)


class WorkerExecutor(Protocol[WorkerRequestContraT, WorkerResponseCoT]):
    async def __call__(
        self, deps: RuntimeDeps, request: WorkerRequestContraT
    ) -> WorkerResponseCoT: ...


@dataclass(frozen=True)
class WorkerCapability(Generic[WorkerRequestT, WorkerResponseT]):
    name: str
    request_model: type[WorkerRequestT]
    response_model: type[WorkerResponseT]
    execute: WorkerExecutor[WorkerRequestT, WorkerResponseT]


class UnknownWorkerCapabilityError(LookupError):
    def __init__(self, capability: str, available_capabilities: tuple[str, ...]) -> None:
        self.capability = capability
        self.available_capabilities = available_capabilities
        super().__init__(
            "Unknown worker capability "
            f"{capability!r}. Available capabilities: {', '.join(available_capabilities)}"
        )
