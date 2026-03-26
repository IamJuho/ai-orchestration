from dataclasses import dataclass
from typing import Protocol

from .config import Settings


class ProviderAdapter(Protocol):
    @property
    def provider_label(self) -> str: ...

    def model_name(self) -> str: ...


@dataclass(frozen=True)
class OpenAIProviderAdapter:
    settings: Settings
    provider_label: str = "openai"

    def model_name(self) -> str:
        return self.settings.model_name


def build_provider_adapter(settings: Settings) -> ProviderAdapter:
    return OpenAIProviderAdapter(settings=settings)
