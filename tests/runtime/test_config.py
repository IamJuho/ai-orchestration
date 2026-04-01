from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_orchestration.deps import RuntimeDeps, build_runtime_deps
from ai_orchestration.runtime.config import Settings
from ai_orchestration.runtime.provider import OpenAIProviderAdapter, build_provider_adapter

ENV_NAMES = [
    "OPENAI_API_KEY",
    "AI_ORCHESTRATION_MODEL",
    "AI_ORCHESTRATION_TIMEOUT_SECONDS",
    "AI_ORCHESTRATION_MAX_STEPS",
    "AI_ORCHESTRATION_RETRY_BUDGET",
]


def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_settings_from_env_loads_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_settings_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings.from_env()

    assert settings.openai_api_key == "test-key"
    assert settings.model_name == "openai:gpt-4o-mini"
    assert settings.timeout_seconds == 30.0
    assert settings.max_steps == 6
    assert settings.retry_budget == 2


def test_settings_direct_init_loads_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings()

    assert settings.openai_api_key == "test-key"
    assert settings.model_name == "openai:gpt-4o-mini"
    assert settings.timeout_seconds == 30.0
    assert settings.max_steps == 6
    assert settings.retry_budget == 2


def test_settings_from_env_supports_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_env(monkeypatch, tmp_path)
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


def test_settings_direct_init_supports_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_env(monkeypatch, tmp_path)
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


def test_settings_from_env_requires_openai_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_env(monkeypatch, tmp_path)

    with pytest.raises(ValidationError):
        Settings.from_env()


def test_settings_direct_init_requires_openai_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_env(monkeypatch, tmp_path)

    with pytest.raises(ValidationError):
        Settings()


def test_settings_from_env_loads_values_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_env(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=dotenv-key",
                "AI_ORCHESTRATION_MODEL=openai:gpt-4o",
                "AI_ORCHESTRATION_TIMEOUT_SECONDS=12.5",
                "AI_ORCHESTRATION_MAX_STEPS=7",
                "AI_ORCHESTRATION_RETRY_BUDGET=1",
            ]
        )
    )

    settings = Settings.from_env()

    assert settings.openai_api_key == "dotenv-key"
    assert settings.model_name == "openai:gpt-4o"
    assert settings.timeout_seconds == 12.5
    assert settings.max_steps == 7
    assert settings.retry_budget == 1


def test_settings_direct_init_loads_values_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_env(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=dotenv-direct-key",
                "AI_ORCHESTRATION_MODEL=openai:gpt-4.1-mini",
                "AI_ORCHESTRATION_TIMEOUT_SECONDS=22",
                "AI_ORCHESTRATION_MAX_STEPS=8",
                "AI_ORCHESTRATION_RETRY_BUDGET=3",
            ]
        )
    )

    settings = Settings()

    assert settings.openai_api_key == "dotenv-direct-key"
    assert settings.model_name == "openai:gpt-4.1-mini"
    assert settings.timeout_seconds == 22.0
    assert settings.max_steps == 8
    assert settings.retry_budget == 3


def test_settings_prefers_existing_environment_over_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AI_ORCHESTRATION_MODEL", "openai:gpt-4o-mini")
    monkeypatch.setenv("AI_ORCHESTRATION_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("AI_ORCHESTRATION_MAX_STEPS", "6")
    monkeypatch.setenv("AI_ORCHESTRATION_RETRY_BUDGET", "2")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=dotenv-key",
                "AI_ORCHESTRATION_MODEL=openai:gpt-4o",
                "AI_ORCHESTRATION_TIMEOUT_SECONDS=12.5",
                "AI_ORCHESTRATION_MAX_STEPS=7",
                "AI_ORCHESTRATION_RETRY_BUDGET=1",
            ]
        )
    )

    settings = Settings.from_env()

    assert settings.openai_api_key == "env-key"
    assert settings.model_name == "openai:gpt-4o-mini"
    assert settings.timeout_seconds == 30.0
    assert settings.max_steps == 6
    assert settings.retry_budget == 2


def test_provider_adapter_uses_settings_model_name() -> None:
    settings = Settings.model_validate(
        {"OPENAI_API_KEY": "test-key", "model_name": "openai:gpt-4o-mini"}
    )

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
