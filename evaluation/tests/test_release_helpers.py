"""Portable environment selection for public evaluation helpers."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "release_candidate_runner", ROOT / "evaluation/run_hunyuan_module_candidate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_runner_defaults_to_current_python(monkeypatch):
    for key in ("HUNYUAN_PYTHON", "METRIC_PYTHON", "HUNYUAN_REPO", "HUNYUAN_MODEL"):
        monkeypatch.delenv(key, raising=False)
    runner = _load_runner()
    assert runner.HUNYUAN_PYTHON == Path(sys.executable)
    assert runner.METRIC_PYTHON == Path(sys.executable)
    assert runner.OFFICIAL == ROOT.parent / "HunyuanVideo-1.5"


def test_candidate_runner_accepts_user_environment(monkeypatch, tmp_path):
    overrides = {
        "HUNYUAN_PYTHON": tmp_path / "generation/bin/python",
        "METRIC_PYTHON": tmp_path / "metrics/bin/python",
        "HUNYUAN_REPO": tmp_path / "upstream",
        "HUNYUAN_MODEL": tmp_path / "weights",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, str(value))
    runner = _load_runner()
    assert runner.HUNYUAN_PYTHON == overrides["HUNYUAN_PYTHON"]
    assert runner.METRIC_PYTHON == overrides["METRIC_PYTHON"]
    assert runner.OFFICIAL == overrides["HUNYUAN_REPO"]
    assert runner.MODEL == overrides["HUNYUAN_MODEL"]
