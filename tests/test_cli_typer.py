from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from local_transcriber.cli import app


def test_print_config_outputs_json(tmp_path: Path) -> None:
    cfg_path = tmp_path / "local-transcriber.toml"
    cfg_path.write_text(
        """
[streaming]
model_size = "base"
""".strip(),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(cfg_path), "--print-config"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert "audio" in payload
    assert "streaming" in payload
    assert payload["streaming"]["model_size"] == "base"
