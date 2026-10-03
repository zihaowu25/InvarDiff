"""Public checkout paths, endpoint ownership and release-file integrity."""

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HUNYUAN_SCRIPTS = (
    "HunyuanVideo/sample_hunyuan.py",
    "HunyuanVideo/sample_hunyuan_step_layer.py",
    "HunyuanVideo/hybrid_cache/sample_hunyuan_magcache_hybrid.py",
    "HunyuanVideo/hybrid_cache/sample_hunyuan_seacache_hybrid.py",
)


def upstream_path(relative):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "OFFICIAL_ROOT"
                for target in node.targets)
    )
    return eval(compile(ast.Expression(assignment.value), str(path), "eval"),
                {"Path": Path, "os": os, "__file__": str(path)})


@pytest.mark.parametrize("relative", HUNYUAN_SCRIPTS)
def test_all_hunyuan_entrypoints_use_the_documented_sibling(relative, monkeypatch):
    monkeypatch.delenv("HUNYUAN_REPO", raising=False)
    assert upstream_path(relative) == (ROOT.parent / "HunyuanVideo-1.5").resolve()


@pytest.mark.parametrize("relative", HUNYUAN_SCRIPTS)
def test_all_hunyuan_entrypoints_accept_an_explicit_checkout(relative, monkeypatch, tmp_path):
    checkout = tmp_path / "upstream"
    monkeypatch.setenv("HUNYUAN_REPO", str(checkout))
    assert upstream_path(relative) == checkout.resolve()


@pytest.mark.parametrize("relative", ("FLUX/sample_flux.py", "FLUX/sample_flux_step_layer.py"))
def test_flux_does_not_override_the_user_hugging_face_endpoint(relative):
    source = (ROOT / relative).read_text()
    assert "HF_ENDPOINT" not in source
    assert "hf-mirror.com" not in source


def test_manifest_sampler_hashes_match_the_packaged_sources():
    manifest = json.loads((ROOT / "cache_books/manifest.json").read_text())
    for item in manifest["policies"].values():
        actual = hashlib.sha256((ROOT / item["script"]).read_bytes()).hexdigest()
        assert actual == item["release_sampler_sha256"]


def test_legacy_private_report_builder_is_not_a_public_entrypoint():
    assert not (ROOT / "evaluation/build_tuning_report.py").exists()


def test_hunyuan_docs_use_current_book_directory_and_name():
    readme = (ROOT / "HunyuanVideo/README.md").read_text()
    assert "| `--cache_book_path` | `cache_books/HunyuanVideo/` |" in readme
    hybrid = (ROOT / "HunyuanVideo/hybrid_cache/README.md").read_text()
    assert "cache_book_hunyuan_<method>_hybrid_" in hybrid
    assert "cache_book_hybrid_<method>_" not in hybrid
