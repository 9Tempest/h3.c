# H3 local inference integration

This standalone harness needs Python, NumPy and standard library modules; it
has no dependency on MLX or a separate kernel benchmark package. It uses
[antirez/h3.c](https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153),
commit `8974cc055ea9c02fcd14cc27dfda3e1027c05153` (MIT), and the original
[MiniMaxAI/MiniMax-H3 FL2VA checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA),
revision `42ed227ee7df40d41602854ae760620d6eb651fe`. Weights retain their own
MiniMax community license; they are downloaded locally, not redistributed.

## Install

From the repository root on an Apple M5 Mac with Xcode, Metal and FFmpeg:

```sh
uv sync --locked
uv run python tools/perf/bootstrap.py
uv run python tools/perf/download.py --workers 16
```

`bootstrap.py` clones the official repository into the separate
`.cache/h3-baseline`, verifies the exact revision and tracked-source digest,
cleans and rebuilds the CLI/library, and builds the session driver. It performs
no GPU execution. Existing dirty or differently pinned sources are rejected;
they are never reset. Build evidence is stored under ignored `.build/`.
Do not pass this report checkout itself as `--source-dir`: new report commits
are intentionally rejected by the runtime baseline guard.

Use `download.py --target /path/to/MiniMax-H3` for a different model location,
then supply `--model-dir /path/to/MiniMax-H3` to each runner. The normal default
is `.cache/models/MiniMax-H3`; no Hugging Face client package is required.

`model-files.json` snapshots the pinned Hugging Face FL2VA tree metadata,
including LFS SHA-256 values. Downloads resume through `.partial` files and
verify hashes before promotion. The complete original family occupies about
144 GB. The text encoder, DiT and VAEs execute in separate stages; download
size is not simultaneous GPU residency. All assets remain under ignored
`.cache/`. Do not run two downloaders against the same destination.

## Benchmark recipe

`fox-winter-t2va-v1` uses this exact prompt:

> A red fox walks through fresh snow in a pine forest. Medium tracking shot,
> natural winter light, realistic fur, soft footsteps and wind.

The default request is 512×512, 5 seconds, 24 fps, seed 42, 20 Euler passes,
all 50 DiT blocks, no velocity/core reuse, no token reduction and no internal
upscaling. This is a low-resolution development request. Twenty steps are the
upstream serving default, not its 50-step quality oracle.

The upstream temporal contract rounds 120 requested frames to 124 frames
(`5 + 17n`), giving 5.1667 seconds. Keep `original.mp4` intact; the runner
separately exports `video-5s.mp4` with exactly 120 frames and verifies both
files with FFprobe. Export time is not charged to denoiser kernel time.

```sh
uv run python tools/perf/run.py --mode bf16 \
  --output-dir results/local/h3/bf16-seed42-001
uv run python tools/perf/run.py --mode upstream-int8 \
  --output-dir results/local/h3/int8-seed42-001
```

Every run needs a new directory. The runner holds the repository's GPU lock
throughout generation and media validation. It records source/binary/model
digests, revisions, environment, request, timestamped raw log offsets, phase
profiles, and output checksums in immutable evidence files. Full model digest
verification precedes timing and can warm the filesystem cache. The result
is not an untouched cold-disk measurement.

The standalone `locking.py` and session driver share the UID-specific temporary
file `m5-max-kernels-<uid>.lock` with the originating experiment tools. This
coordinates cooperating processes only. The runtime compiles Metal shaders
when it starts; a successful CPU build is not GPU validation.

Use `--gpu-wait-seconds 1800` to queue behind another local experiment after
model verification. The default is an immediate lock check. Queue time is
recorded separately and excluded from generation process time; the acquired
lock stays held through generation and media validation.

`bf16` explicitly disables M5's default INT8 QKV, MLP and attention-output
projections. It is the upstream BF16 close-reference path, not a proof of
parity with the official Python pipeline. `upstream-int8` retains upstream
grouped FC2 and its existing fusion; it is not this repository's per-row
INT8 or Hadamard candidate. MXINT8 and FP8 are not substituted silently.
TensorOps compilation failure is recorded as unsupported and the automatic
upstream fallback is rejected. Both modes explicitly use reference RoPE.

## Evidence interpretation

One process includes runtime shader compilation, loading, weight preparation,
text encoding, denoising, audio/video VAE and MP4 creation. `--profile` reports
component phase walls, encode/wait counters, root GPU duration and allocation
counters. A component's `total` overlaps its phase marks; do not sum both.
Tracked tensor peak is not process RSS or whole-machine memory consumption.

The 124-frame geometry has 37 video latent timesteps. At 512 square this is
9472 video tokens + 414 stereo-audio tokens + actual text tokens. The initial
formula is source-derived, not an observed full runtime shape; use the optional
diagnostic patch to confirm real packed geometry. Existing T=4096 synthetic
kernel results cannot be multiplied into this request's latency.

Media structure validation establishes decodability, dimensions, frame count,
and duration. It does not establish numerical parity or perceptual quality.
Compare candidate outputs against the same seed/configuration BF16 output,
inspect motion and audio, and evaluate another seed. Any speedup promotion
requires independent reruns, synchronized warm timings and numerical/quality
checks, including online activation transformation and quantization overhead.

Candidate work is deliberately separate: INT8 + offline-weight/online-activation
Hadamard, standalone FP8, attention/layout/fusion, and resident-session caching.
No FP8 + Hadamard deployment claim is made.

The separate [native FC2 Hadamard experiment](experimental-hadamard.md) is
explicitly opt-in and uses `.cache/h3-hadamard`. Its patch, capture runner,
independent projection readback driver and NumPy analyzer are included for
reproduction. Synthetic projection correctness is distinct from the pending
real-activation, full-trajectory and perceptual gates. It does not modify the
clean baseline or authorize replacing the original checkpoint.

## Strict evaluation and progress analysis

```sh
uv run python tools/perf/evaluate.py \
  results/local/h3/bf16-seed42-001/video-5s.mp4 \
  results/local/h3/int8-seed42-001/video-5s.mp4 \
  --kind media --ssim --output-dir results/local/h3/eval-seed42-001
uv run python tools/perf/analyze_progress.py \
  results/local/h3/bf16-seed42-001 \
  --output results/local/h3/bf16-progress-001.json
```

`evaluate.py` checks matching video/audio timing and shapes before decoding;
it reports RGB and PCM SNR, PSNR, relative L2, RMSE, maximum error, finite
counts, and optional planar-RGB FFmpeg SSIM. Array input uses
`--kind arrays --data-range VALUE` and supports `.npy`/`.npz` under its explicit
contract. Results distinguish undefined/infinite metrics without JSON NaN.
LPIPS/VBench adapters are not configured and remain `not_measured`; FVD is
not computed from one pair. Quality remains `not_evaluated`. Run a BF16
self-comparison to verify zero-error behavior before interpreting drift.
See the [evaluation protocol](evaluation.md) for metric definitions, alignment
requirements, research sources and the separate quality promotion criteria.

`analyze_progress.py` validates source recipe, log SHA and complete byte-offset
coverage, excludes the first step, and excludes ambiguous/coalesced receipt
intervals. Its median and raw values describe log receipt timing only, not
compilation-free latency or GPU time. Output JSON must be a new file.

`compare.py` is retained as a legacy contact-sheet/basic comparison helper;
use `evaluate.py` for published strict alignment and metrics. The measured pair,
metric definitions and limitations are in the [performance report](../../docs/performance/m5-max-512-5s/README.md).

## CPU-only validation

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The configured suite tests source guards, downloads with mocked HTTP,
diagnostic validation, progress analysis, metric edge cases and standalone CLI
imports. It launches no model inference or Metal work. GPU validation remains
unavailable on a non-Metal host; passing these tests is not a GPU result.

The added harness and tests retain their MIT attribution in [LICENSE](LICENSE).
Upstream runtime attribution is retained by the root license and
[UPSTREAM-LICENSE](UPSTREAM-LICENSE).

## Resident requests and diagnostics

`session.c` is a small driver using the pinned public H3 API. `bootstrap.py`
builds it against the unmodified runtime. Measure one complete warmup and
three cached generations with the same recipe:

```sh
uv run python tools/perf/run_session.py --mode upstream-int8 \
  --output-dir results/local/h3/int8-cached-seed42-001
```

The C driver owns the GPU lock for the complete session; do not wrap it in a
second lock. The Python wrapper rejects nonzero `--gpu-wait-seconds` and `--diagnostics`.
Audio VAE context creation and loading still repeat, so cached end-to-end
latency must not be labelled compilation-free. The wrapper reports raw
generation times and separately instrumented synchronized denoise callback
times. Upstream cached DiT phase-wall marks span intervening decode/export
work and cannot be used as per-generation denoise measurements.

The optional [diagnostic build](diagnostics.md) records actual geometry,
selected branches and sampled post-Euler finite statistics. Keep its overhead
separate from baseline timing and verify same-seed output equivalence.
Build it with `uv run python tools/perf/bootstrap.py --diagnostics`, then pass
`--diagnostics --source-dir .cache/h3-diagnostics` to the one-shot runner.

`download_ranges.py` is an optional, independently tested HTTP Range helper
for a remaining single-file download. Stop the original downloader and all
its children first. It preserves the existing prefix, validates response
ranges and final pinned SHA256, and retains its extra chunks for inspection.
Its tests use mocked HTTP; throughput improvement is not established.
