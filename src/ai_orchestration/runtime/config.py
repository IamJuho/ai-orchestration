import os

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    openai_api_key: str = Field(alias="OPENAI_API_KEY", min_length=1)
    model_name: str = "openai:gpt-4o-mini"
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_steps: int = Field(default=6, ge=1)
    retry_budget: int = Field(default=2, ge=0)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.model_validate(
            {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "model_name": os.getenv("AI_ORCHESTRATION_MODEL", "openai:gpt-4o-mini"),
                "timeout_seconds": os.getenv("AI_ORCHESTRATION_TIMEOUT_SECONDS", 30.0),
                "max_steps": os.getenv("AI_ORCHESTRATION_MAX_STEPS", 6),
                "retry_budget": os.getenv("AI_ORCHESTRATION_RETRY_BUDGET", 2),
            }
        )
