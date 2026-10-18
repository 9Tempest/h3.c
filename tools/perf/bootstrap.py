"""Clone and build the pinned runtime separately from this reporting fork; never run Metal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess

import run as h3


REPOSITORY = "https://github.com/antirez/h3.c.git"
BASELINE_SOURCE_SHA256 = "6e8931b5d88abfc40ea4f38c3c2c603a5b1ae93156f19b653cbfe9f26c8814b5"


def command(argv):
    subprocess.run([str(value) for value in argv], check=True)


def prepare_source(source, diagnostics=False):
    source = source.resolve()
    if source == h3.ROOT.resolve():
        raise ValueError("runtime must be a separate pinned checkout, not this reporting fork")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", "--no-checkout", REPOSITORY, source])
        command(["git", "-C", source, "checkout", "--detach", h3.UPSTREAM_REVISION])
    provenance = h3.git_provenance(source)
    if diagnostics and not provenance["dirty"]:
        h3.validate_upstream(source, provenance)
        if provenance["source_sha256"] != BASELINE_SOURCE_SHA256:
            raise ValueError("clean runtime tracked-source digest differs from the pinned source")
        if source == (h3.ROOT / ".cache/h3-baseline").resolve():
            raise ValueError("diagnostics cannot modify the default baseline checkout")
        patch = Path(__file__).with_name("diagnostics.patch").resolve()
        if h3.sha256(patch) != h3.DIAGNOSTICS_PATCH_SHA256:
            raise ValueError("diagnostic patch digest differs from the reviewed patch")
        command(["git", "-C", source, "apply", "--check", patch])
        command(["git", "-C", source, "apply", patch])
        provenance = h3.git_provenance(source)
    h3.validate_upstream(source, provenance, diagnostics)
    if not diagnostics and provenance["source_sha256"] != BASELINE_SOURCE_SHA256:
        raise ValueError("clean runtime tracked-source digest differs from the pinned source")
    return source, provenance


def build_commands(source, jobs, session):
    commands = [
        ["make", "-C", str(source), "clean"],
        ["make", "-C", str(source), f"-j{jobs}", "all"],
    ]
    if session:
        commands.append(
            [
                "clang",
                "-O2",
                "-std=c11",
                "-D_DARWIN_C_SOURCE",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(source),
                str(Path(__file__).with_name("session.c")),
                str(source / "libh3.a"),
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-framework",
                "MetalPerformanceShaders",
                "-framework",
                "MetalPerformanceShadersGraph",
                "-framework",
                "Accelerate",
                "-licucore",
                "-lm",
                "-o",
                str(h3.ROOT / ".build/h3-session"),
            ]
        )
    return commands


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path)
    cli.add_argument("--diagnostics", action="store_true")
    cli.add_argument("--jobs", type=int, default=8)
    return cli


def main():
    args = parser().parse_args()
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise h3.UnsupportedError("building this runtime requires Apple Silicon and Xcode")
    source = args.source_dir or h3.ROOT / (
        ".cache/h3-diagnostics" if args.diagnostics else ".cache/h3-baseline"
    )
    source, provenance = prepare_source(source, args.diagnostics)
    build = h3.ROOT / ".build"
    build.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    evidence = build / f"bootstrap-{timestamp}.json"
    record = dict(
        status="failed",
        upstream=provenance,
        diagnostics=args.diagnostics,
        gpu_execution="not performed",
        commands=[],
    )
    try:
        for argv in build_commands(source, args.jobs, not args.diagnostics):
            record["commands"].append(argv)
            command(argv)
        h3.validate_upstream(source, h3.git_provenance(source), args.diagnostics)
        record["compiler"] = h3.read_command(["clang", "--version"])
        record["binary_sha256"] = h3.sha256(source / "h3")
        record["library_sha256"] = h3.sha256(source / "libh3.a")
        if not args.diagnostics:
            record["session_sha256"] = h3.sha256(build / "h3-session")
        record["status"] = "built"
    except Exception as error:
        record["error"] = str(error)
        raise
    finally:
        h3.write_json(evidence, record)
    print(f"Built pinned runtime; no GPU execution. Evidence: {evidence}")


if __name__ == "__main__":
    main()
