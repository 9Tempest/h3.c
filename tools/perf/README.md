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
reproduction. `fc2_campaign.py` reproduces both synthetic correctness and
frozen real-activation calibration under the shared GPU lock. Real FC2
aggregate gains and an explicit tail regression are reported in the
[follow-up](../../docs/performance/m5-max-512-5s/followup.md); full-model adoption
remains a separate decision. It does not modify the clean baseline or replace weights.
The later two-seed model evidence and its limits are in the
[native follow-up](../../docs/performance/m5-max-512-5s/native-experiments.md).

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


## Prepared checkpoints and native experiments

The second release includes the prepared checkpoint exporter/importer, frozen
fusion campaign and full-model capture runners. Each verifies its own source
and patch digests; every runtime is a separate checkout. The following table
is the source contract, not permission to mix patches:

| Runtime directory | Patch applied to pinned upstream | Purpose |
| --- | --- | --- |
| `.cache/h3-baseline` | none | Original clean BF16/upstream INT8 |
| `.cache/h3-hadamard` | `hadamard.patch` | On-load H256, original-model captures/export |
| `.cache/h3-prepared` | `prepared-checkpoint.patch` | Import previously verified derived weights |
| `.cache/h3-fused` | `fused.patch` | On-load H256 with opt-in gate epilogues |

All three patches include the H256/capture changes; do not apply them on top
of each other. To create an experimental runtime, use a new directory,
checkout `8974cc055ea9c02fcd14cc27dfda3e1027c05153`, apply the selected patch,
and run `make -C DIRECTORY -j8 all`. For example:

```sh
git clone --no-checkout https://github.com/antirez/h3.c.git .cache/h3-prepared
git -C .cache/h3-prepared checkout --detach 8974cc055ea9c02fcd14cc27dfda3e1027c05153
git -C .cache/h3-prepared apply "$PWD/tools/perf/prepared-checkpoint.patch"
make -C .cache/h3-prepared -j8 all
```

The runners reject unexpected tracked changes, untracked source files, wrong
pins and automatic precision fallback. Successful build-only validation does
not establish that a GPU experiment passes. Keep generated files outside the
runtime checkout.

Create the H256 runtime similarly using `hadamard.patch`, then export and
assemble a derived checkpoint without deleting or rewriting the original:

```sh
uv run python tools/perf/prepared_checkpoint.py \
  --source-dir .cache/h3-hadamard --output-dir .cache/checkpoints/h3-h256
uv run python tools/perf/derived_checkpoint.py \
  --checkpoint-dir .cache/checkpoints/h3-h256
```

These commands perform offline GPU weight preparation and substantial disk
I/O under the shared lock. Exported weights are local artifacts covered by
the model's license; no checkpoint payload is included in this repository.
Use the `derived_manifest_sha256` emitted by the assembly record for later
`--checkpoint-manifest-sha256` arguments. `run_prepared_checkpoint.py --help`
describes replay against an existing on-load H256 capture. Prepared loading
plus fusion is not a supported combined runtime in this release.

`benchmarks.json` fixes the person-turn, rainy-street and water-pour prompts
and seeds. For each benchmark, run BF16 from the H256 source with the original
model, then prepared H256 from the derived checkpoint, preserving separate
output directories. For example:

```sh
uv run python tools/perf/run_benchmark.py --benchmark-id person-turn-v1 \
  --mode bf16 --output-dir results/local/h3/person-bf16-NEW
uv run python tools/perf/run_benchmark.py --benchmark-id person-turn-v1 \
  --mode hadamard-fc2 --checkpoint-dir .cache/checkpoints/h3-h256 \
  --checkpoint-manifest-sha256 YOUR_DERIVED_MANIFEST_SHA256 \
  --output-dir results/local/h3/person-h256-NEW
uv run python tools/perf/evaluate_benchmark_pair.py \
  --bf16 results/local/h3/person-bf16-NEW \
  --hadamard results/local/h3/person-h256-NEW --media \
  --output-dir results/local/h3/person-evaluation-NEW
```

`run_long_benchmark.py` fixes the separate full Chinese prompt, image hash,
seed83 and 512×288/15-second contract in `long_benchmark.json`. Its
`--input-image` must match the pinned input hash; the private source JPEG is
not redistributed, so exact image-conditioned reproduction requires access
to that original image. The runner checks its tokenizer result, conditioning
geometry and both VAE config pins before inference. `evaluate_long_pair.py`
retains 362 native frames while comparing separate exact 360-frame exports.
It rejects media or latent shape/timing mismatches instead of aligning them
silently.

Build `.cache/h3-fused` from `fused.patch` to use `run_fused_campaign.py` for
independent full-K references, numerical parity and warmed operator timings.
`run_fused_experiment.py` captures a complete fox request and checks it
against `--reference-run` from on-load H256. `run_native_session.py` accepts
`--source-dir .cache/h3-fused --fusion none|fc2|attention|both`; it holds the
shared lock and saves exact harness snapshots for one warmup plus three
same-seed requests. Use the same source and policy for paired performance
runs. Both flags default to off. Fresh synchronized denoise callbacks and
cached generation times have different scopes; repeated audio decoder setup
precludes a compilation-free full-generation claim.

The portable distribution changes local directory/import plumbing from the
originating experiment repository. Its harness digests therefore differ from
historical reports; it retains the exact pinned runtime patches and numerical
contracts. Do not relabel a historical record with a portable harness hash.


The [fusion protocol](fused.md) records the full numerical contract and measured
operator/resident results. `compare_resident.py` validates original local run
evidence against the exact portable harness pins; it does not accept historical
main-repository records with different harness hashes. Its exported timing
comparison is descriptive and retains separate raw arrays and process order.

```sh
uv run python tools/perf/compare_resident.py \
  --baseline-run results/local/h3/fused-none-NEW/run.json \
  --candidate-run results/local/h3/fused-attention-NEW/run.json \
  --output-dir results/local/h3/resident-comparison-NEW
```

[Distribution source digests](distribution-provenance.json) distinguish unchanged
runtime patches from the portable harness adaptation.
