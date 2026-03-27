import pytest
from pydantic import ValidationError

from ai_orchestration.deps import RuntimeDeps, build_runtime_deps
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.provider import OpenAIProviderAdapter, build_provider_adapter


def test_settings_from_env_loads_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AI_ORCHESTRATION_MODEL", raising=False)
    monkeypatch.delenv("AI_ORCHESTRATION_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("AI_ORCHESTRATION_MAX_STEPS", raising=False)
    monkeypatch.delenv("AI_ORCHESTRATION_RETRY_BUDGET", raising=False)

    settings = Settings.from_env()

    assert settings.openai_api_key == "test-key"
    assert settings.model_name == "openai:gpt-4o-mini"
    assert settings.timeout_seconds == 30.0
    assert settings.max_steps == 6
    assert settings.retry_budget == 2


def test_settings_direct_init_loads_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AI_ORCHESTRATION_MODEL", raising=False)
    monkeypatch.delenv("AI_ORCHESTRATION_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("AI_ORCHESTRATION_MAX_STEPS", raising=False)
    monkeypatch.delenv("AI_ORCHESTRATION_RETRY_BUDGET", raising=False)

    settings = Settings()

    assert settings.openai_api_key == "test-key"
    assert settings.model_name == "openai:gpt-4o-mini"
    assert settings.timeout_seconds == 30.0
    assert settings.max_steps == 6
    assert settings.retry_budget == 2


def test_settings_from_env_supports_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AI_ORCHESTRATION_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("AI_ORCHESTRATION_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("AI_ORCHESTRATION_MAX_STEPS", "9")
    monkeypatch.setenv("AI_ORCHESTRATION_RETRY_BUDGET", "4")

    settings = Settings.from_env()

    assert settings.model_name == "openai:gpt-4o"
    assert settings.timeout_seconds == 12.5
    assert settings.max_steps == 9
    assert settings.retry_budget == 4


def test_settings_direct_init_supports_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AI_ORCHESTRATION_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("AI_ORCHESTRATION_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("AI_ORCHESTRATION_MAX_STEPS", "9")
    monkeypatch.setenv("AI_ORCHESTRATION_RETRY_BUDGET", "4")

    settings = Settings()

    assert settings.model_name == "openai:gpt-4o"
    assert settings.timeout_seconds == 12.5
    assert settings.max_steps == 9
    assert settings.retry_budget == 4


def test_settings_from_env_requires_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings.from_env()


def test_settings_direct_init_requires_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings()


def test_provider_adapter_uses_settings_model_name() -> None:
    settings = Settings.model_validate({"OPENAI_API_KEY": "test-key"})

    provider = build_provider_adapter(settings)

    assert isinstance(provider, OpenAIProviderAdapter)
    assert provider.provider_label == "openai"
    assert provider.model_name() == "openai:gpt-4o-mini"


def test_build_runtime_deps_wires_config_provider_and_label() -> None:
    settings = Settings.model_validate({"OPENAI_API_KEY": "test-key"})

    deps = build_runtime_deps(config=settings)

    assert isinstance(deps, RuntimeDeps)
    assert deps.config is settings
    assert isinstance(deps.provider, OpenAIProviderAdapter)
    assert deps.provider_label == "openai"
    assert deps.trace_events == []
