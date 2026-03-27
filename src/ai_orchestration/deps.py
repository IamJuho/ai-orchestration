from dataclasses import dataclass, field

from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.provider import ProviderAdapter, build_provider_adapter
from ai_orchestration.state.base import StateStore


@dataclass
class RuntimeDeps:
    config: Settings
    provider: ProviderAdapter
    provider_label: str
    state_store: StateStore | None = None
    trace_events: list[dict[str, str]] = field(default_factory=list)


def build_runtime_deps(config: Settings, state_store: StateStore | None = None) -> RuntimeDeps:
    provider = build_provider_adapter(config)
    return RuntimeDeps(
        config=config,
        provider=provider,
        provider_label=provider.provider_label,
        state_store=state_store,
    )
