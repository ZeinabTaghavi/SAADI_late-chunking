from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "run_qasper_musique_c250_overlaps_gpu0123.sh"
)


def test_overlap_launcher_dry_run_covers_all_expected_configurations(
    tmp_path: Path,
):
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "OUTPUT_ROOT": str(tmp_path / "late_chunk_runs"),
            "EVALUATION_ROOT": str(tmp_path / "late_chunk_evaluations"),
            "LOG_DIR": str(tmp_path / "logs"),
            "TABLE_OUTPUT": str(tmp_path / "docs" / "results.tex"),
            "TABLE_JSON_OUTPUT": str(tmp_path / "docs" / "results.json"),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "OVERLAPS_TO_ENSURE=0 25 50" in result.stdout
    assert "TOTAL_CONFIGURATIONS_TO_ENSURE=60" in result.stdout
    assert "[1/60] dataset=qasper overlap=0 retriever=jina-v3" in result.stdout
    assert (
        "[60/60] dataset=musique_4hop overlap=50 retriever=bge-m3"
        in result.stdout
    )
    assert "--chunk-overlap 25" in result.stdout
    assert "--chunk-overlap 50" in result.stdout
    assert "--max-docs 25" in result.stdout
