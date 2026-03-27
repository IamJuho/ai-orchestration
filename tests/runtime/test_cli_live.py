import json
import os
import subprocess
import sys

import pytest

from ai_orchestration.runtime.runner import ExitStatus


@pytest.mark.live
def test_cli_live_json_smoke_response_shape() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_orchestration.runtime.cli",
            "--objective",
            "Live smoke verification",
            "--json",
            "--max-steps",
            "1",
            "--timeout-seconds",
            "45",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    payload = json.loads(completed.stdout)

    assert set(payload) == {"run_id", "status", "summary", "artifacts", "failures"}
    assert isinstance(payload["run_id"], str)
    assert payload["run_id"]
    assert payload["status"] in {status.value for status in ExitStatus}
    assert isinstance(payload["summary"], str)
    assert isinstance(payload["artifacts"], list)
    assert isinstance(payload["failures"], list)

    for artifact in payload["artifacts"]:
        assert set(artifact).issuperset({"artifact_id", "kind", "producer", "content", "metadata"})
    for failure in payload["failures"]:
        assert set(failure).issuperset({"kind", "message", "retryable", "details"})

    if payload["status"] == ExitStatus.SUCCESS.value:
        assert completed.returncode == 0
    else:
        assert completed.returncode == 1
