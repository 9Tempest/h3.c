"""Smoke tests of the standalone distribution, without models or GPU calls."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import bootstrap
import run


DIRECTORY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "bootstrap.py",
        "run.py",
        "run_session.py",
        "evaluate.py",
        "analyze_progress.py",
        "download.py",
        "download_ranges.py",
        "compare.py",
        "run_fc2_experiment.py",
        "fc2_analyze.py",
        "fc2_campaign.py",
        "prepared_checkpoint.py",
        "derived_checkpoint.py",
        "run_prepared_checkpoint.py",
        "evaluate_latents.py",
        "evaluate_benchmark_pair.py",
        "evaluate_long_pair.py",
        "run_benchmark.py",
        "run_long_benchmark.py",
        "run_fused_campaign.py",
        "run_fused_experiment.py",
        "run_native_session.py",
        "compare_resident.py",
        "publish_resident.py",
        "publish_samples.py",
        "publish_fused_campaign.py",
        "publish_fused_model_parity.py",
        "compare_fused_campaigns.py",
    ],
)
def test_help_is_standalone(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(DIRECTORY / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_baseline_is_not_the_reporting_checkout(tmp_path):
    args = run.parser().parse_args(["--mode", "bf16", "--output-dir", str(tmp_path)])
    assert args.source_dir == run.ROOT / ".cache/h3-baseline"
    with pytest.raises(ValueError, match="separate pinned checkout"):
        bootstrap.prepare_source(run.ROOT)


def test_new_fork_revision_is_not_accepted_as_runtime(tmp_path):
    with pytest.raises(ValueError, match="clean at the pinned revision"):
        run.validate_upstream(tmp_path, {"revision": "report-commit", "dirty": False})


def test_bootstrap_commands_only_compile(tmp_path):
    commands = bootstrap.build_commands(tmp_path, 8, True)
    assert commands[0] == ["make", "-C", str(tmp_path), "clean"]
    assert commands[1] == ["make", "-C", str(tmp_path), "-j8", "all"]
    assert commands[2][0] == "clang"
    assert all(argv[0] in ("make", "clang") for argv in commands)
    assert "test" not in commands[1]


def test_shared_lock_name_is_unchanged():
    assert 'f"m5-max-kernels-{os.getuid()}.lock"' in (DIRECTORY / "locking.py").read_text()
    assert "m5-max-kernels-%" in (DIRECTORY / "session.c").read_text()


def test_published_samples_match_summary():
    report = run.ROOT / "docs/performance/m5-max-512-5s"
    summary = json.loads((report / "sanitized_summary.json").read_text())
    for item in summary["runs"]:
        path = report / item["sample"]["file"]
        assert path.stat().st_size == item["sample"]["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sample"]["sha256"]
    assert summary["evaluation"]["quality_decision"] == "not_evaluated"
    assert summary["measurement_scope"]["independent_rerun"] == "not measured"
