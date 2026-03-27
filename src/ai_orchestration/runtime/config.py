import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    openai_api_key: str = Field(alias="OPENAI_API_KEY", min_length=1)
    model_name: str = "openai:gpt-4o-mini"
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_steps: int = Field(default=6, ge=1)
    retry_budget: int = Field(default=2, ge=0)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def _apply_environment_defaults(cls, data: Any) -> Any:
        if data is None:
            values: dict[str, Any] = {}
        elif isinstance(data, Mapping):
            values = dict(data)
        else:
            return data

        if "OPENAI_API_KEY" not in values and "openai_api_key" not in values:
            env_api_key = os.getenv("OPENAI_API_KEY")
            if env_api_key is not None:
                values["OPENAI_API_KEY"] = env_api_key

        if "model_name" not in values:
            env_model_name = os.getenv("AI_ORCHESTRATION_MODEL")
            if env_model_name is not None:
                values["model_name"] = env_model_name

        if "timeout_seconds" not in values:
            env_timeout = os.getenv("AI_ORCHESTRATION_TIMEOUT_SECONDS")
            if env_timeout is not None:
                values["timeout_seconds"] = env_timeout

        if "max_steps" not in values:
            env_max_steps = os.getenv("AI_ORCHESTRATION_MAX_STEPS")
            if env_max_steps is not None:
                values["max_steps"] = env_max_steps

        if "retry_budget" not in values:
            env_retry_budget = os.getenv("AI_ORCHESTRATION_RETRY_BUDGET")
            if env_retry_budget is not None:
                values["retry_budget"] = env_retry_budget

        return values

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()
