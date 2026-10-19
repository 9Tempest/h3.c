# M5 Max: a 512 × 512, five-second H3 inference observation

On one Apple M5 Max, the pinned upstream INT8 path completed this request in
**182.455 seconds**, versus **252.938 seconds** with the upstream BF16 projection
overrides. The observed process-time ratio is **1.386×**. Most of the difference
was in DiT denoising: **126.556 versus 205.297 seconds**. Video VAE decoding stayed
near **30 seconds**. These are one process per mode, one seed, and a fixed run
order; they do not establish a stable speedup or a quality acceptance decision.

The [follow-up evidence](followup.md) adds real FC2 activation comparisons,
actual packed geometry and a partial Metal trace. It preserves the initial
paired process measurement and treats diagnostic captures separately.
The [native experiment report](native-experiments.md) organizes the later
two-seed latent/media results, prepared-checkpoint replay, resident timings and
two gate-epilogue campaigns, three additional scenes and the matched resident
attention-fusion result. Its single-seed timing ratios remain descriptive.

This fork adds an independently usable [performance and evaluation harness](../../../tools/perf/README.md).
The measured runtime was the clean, unmodified
[antirez/h3.c commit 8974cc0](https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153).
The harness builds that commit in a separate checkout and rejects a changed
runtime revision, including this fork's report commit. The portable tools were
adapted after measurement; original runner, source, executable, model and evidence
hashes are retained in [sanitized_summary.json](sanitized_summary.json).

## Request, hardware and precision

The benchmark is `fox-winter-t2va-v1`:

> A red fox walks through fresh snow in a pine forest. Medium tracking shot,
> natural winter light, realistic fur, soft footsteps and wind.

Both modes used seed **42**, **512 × 512**, **24 fps**, **20 Euler passes**, all
**50 DiT blocks**, reference RoPE, reuse intervals of **1**, no token reduction
and no internal upscaling. Twenty passes follow the upstream serving default;
this is not a 50-pass quality reference. The 120-frame request becomes **124
native frames (5.1667 seconds)** under the upstream temporal contract. The
original MP4 is preserved locally, and a separate exact five-second sample is
trimmed and re-encoded to **120 frames** for delivery and evaluation.

The model is the original
[MiniMaxAI/MiniMax-H3 FL2VA checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA),
revision `42ed227ee7df40d41602854ae760620d6eb651fe`, approximately **144 GB** of
persistent files. Its weights and license remain with MiniMax; no weights are
redistributed here. The runtime loads stages separately, so download size is
not simultaneous GPU residency.

Hardware/software recorded on 2026-10-18: **Apple M5 Max**, **128 GiB unified
memory**, Darwin **27.0.0**, arm64, **Xcode 27.1 (27A9275)**, Apple clang **21.0.0**,
Python **3.12.15**, FFmpeg/FFprobe **9.0.2**. The evaluation environment reported
macOS 27.0.1 and NumPy 2.4.4. These are recorded environment strings, not a
requirement to reproduce identical timing on a different machine.

`bf16` enables all three `--use-slower-bf16-*` overrides for DiT QKV, MLP and
attention-output projections. `upstream-int8` uses upstream's on-load INT8 weights,
online activation quantization, grouped FC2 and existing fusions. The remaining
pipeline has the upstream mixed floating-point behavior, including BF16 final
heads. Neither name claims a uniformly typed whole model. The BF16 output is a
close reference within h3.c, not established parity with the official Python
pipeline. Unsupported TensorOps compilation/fallback is rejected by the runner.

## Observed time and memory

| Measurement | BF16 overrides | Upstream INT8 |
| --- | ---: | ---: |
| Generation process wall time | 252.938 s | 182.455 s |
| Qwen text encoder total | 5.355 s | 4.761 s |
| DiT load and preparation | 8.707 s | 19.633 s |
| DiT Euler denoise phase wall | 205.297 s | 126.556 s |
| Audio VAE decoder total | 2.165 s | 0.473 s |
| Video VAE decoder total | 30.218 s | 30.148 s |
| DiT tracked tensor peak | 37.347 GiB | 19.812 GiB |
| Median progress-receipt interval, excluding step 0 | 10.257863 s | 6.324248 s |

The process timer includes runtime shader compilation, model loading, weight
preparation, text encoding, denoising, VAEs and original MP4 export. The later
five-second trim/re-encode and evaluation are outside that timer. Full model
hash verification precedes timing and can warm the filesystem cache. BF16 ran
first and INT8 second; first-use compilation, OS/driver caches, thermal state and
run order were not controlled by a repeated, interleaved experiment. For example,
the audio VAE difference cannot be attributed to DiT INT8 arithmetic.

Phase totals and subphases overlap and must not be added together. The DiT
tracked tensor peak excludes MPS internal allocations, CPU and media buffers;
it is not process RSS or system peak memory. INT8 reduces that tracked peak but
adds **10.926 seconds** of observed DiT preparation in this pair. Online
activation quantization is included in denoising and has not been isolated.

The 19 post-first-step intervals are descriptive **log receipt** measurements.
The analyzer validates immutable log hashes, full byte-offset coverage and
unambiguous progress boundaries. In the pinned default M5 window of one
evaluation, each completed enqueue boundary follows a tracked command-buffer
submit/wait. Parent-process pipe reading and log flushing add unknown delay.
Removing step 0 does not prove all compilation finished, and later Euler steps
are different inputs rather than repeated identical work. These intervals are
neither compilation-free kernel latency nor device timestamps. Raw intervals
and the original counter fields are in the summary.

The upstream BF16 denoise counters report 167.986 s in `encode` and only 0.104 s
in `root-gpu`; INT8 reports 0.085 s and 1.135 s, respectively. **This does not
establish a CPU bottleneck or device operation ranking.** MPSGraph encoding can
include compilation, allocation, scheduling and waiting; root command-buffer
GPU duration can omit MPS child buffers. Source counters show 1,000 attention
calls for both modes, BF16 4,000 MPS-linear calls, and INT8 8,200 direct dispatch
counts. These counters have different scopes and are not interchangeable with
shader launches or GPU time.

## Output drift and samples

- [BF16 five-second sample](bf16-seed42.mp4)
- [Upstream INT8 five-second sample](int8-seed42.mp4)

The strict evaluator checks frame/sample counts and exact PTS alignment before
comparison. Both samples contain **120 frames at 512 × 512 / 24 fps** and
**160,000 stereo audio samples at 32 kHz**. It decodes RGB uint8 and PCM float32,
accumulates numeric metrics in float64, checks finite values, and does not
silently align, truncate or resample unequal inputs.

| Paired drift metric | INT8 versus BF16 |
| --- | ---: |
| RGB SNR | 12.712753 dB |
| RGB PSNR, range 255 | 18.629431 dB |
| RGB relative L2 | 0.23139946 |
| RGB SSIM, FFmpeg planar RGB | 0.690500567 |
| Stereo PCM SNR | 13.961846 dB |
| Stereo PCM relative L2 | 0.20040461 |
| Stereo PCM PSNR, range 2 | 61.356032 dB |

These are **decoded output drift** measurements after video/audio compression
and five-second export, not isolated quantization error or a perceptual score
against ground truth. RGB SSIM is explicitly the planar RGB FFmpeg variant;
YUV SSIM or a differently configured implementation is not the same metric.
The PCM PSNR uses full-scale range 2; the clips have much smaller amplitudes,
so that number should not be read as negligible audio drift. BF16 self-comparison
has zero numeric error and SSIM 1, validating the comparison plumbing only.

The result remains **quality not evaluated**. No pass threshold was chosen after
seeing these outputs. LPIPS and VBench are **not measured**, and FVD needs a
collection of clips rather than one pair. A quality decision needs matched
prompts/seeds, human checks of motion, detail and audio, and an appropriate
dataset-level protocol; it must be separate from implementation correctness.

## Source-backed bottleneck hypotheses

The measured phase bottleneck is DiT denoising; the next measured stage is video
VAE. The following finer-grained explanations are source-backed hypotheses,
not per-operation device measurements. The packed shape is **9,914 rows**:
9,472 video, 414 stereo-audio and 28 prompt rows. The prompt count was checked
separately with the pinned CPU tokenizer. The baseline logs did not capture
every operation's actual geometry.

| Area | What the pinned source establishes | Experiment and acceptance evidence |
| --- | --- | --- |
| BF16 large-row projections | Native BF16 TensorOps selection has row thresholds; automatic chunking covers 2,049–3,072 rows, while this request has 9,914. The default QKV/output path falls through to MPSGraph. Default `H3_NAX=qkv-attn` also leaves BF16 MLP on its MPSGraph path. | Test an isolated 2,048-row split and suitable native BF16 MLP mode, including tail 1,722, graph/launch costs and output drift. Preserve the default baseline and record unsupported variants. |
| Attention and layouts | Attention uses cached noncausal MPSGraph SDPA. BF16 graph construction transposes Q/K/V and output; physical copies are a compiler/runtime question. Q/K/V each have about 135.5 MiB at this shape. | GPU Capture with verified mapping to operation groups; inspect actual layout materialization and memory traffic, then evaluate head-major/fused or streaming SDPA variants against an independent reference. Do not assume a dense score matrix is explicitly allocated. |
| INT8 FFN/FC2 | Upstream FC2 uses 14 groups of 1,024, pads 9,914 rows to 9,984 and selects the large-row local quantizer. It is not a single-scale per-row FC2 implementation. | Capture real post-SwiGLU activations and measure online quantization, GEMM and fused end-to-end costs. A block-256 Hadamard transform changes the quantization contract and adds runtime work; keep implementation and quantization error separate. |
| Load and reuse | INT8 preparation is more expensive in this pair. The public session API can retain prompt embeddings, DiT and video decoder; audio VAE context/load still repeat. | One full warmup plus at least three cached generations, raw synchronized times, seed parity and controlled order. Any saved quantized-weight cache needs revision/layout/device keys and separate offline preparation accounting. |
| Video VAE | At 512 square, the 288-pixel tile policy gives 2 × 2 spatial tiles and seven temporal chunks: 28 tile forwards and 1,008 attention calls. Both modes spend about 30 s here. | Compare supported tile policies, memory reuse and precision/fusion choices, checking overlap seams, temporal edges, finite values and output quality. Include decode and stitching costs. |

Relevant pinned source: [BF16 linear dispatch](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L2443),
[QKV dispatch](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L3798),
[SDPA graph](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L1448),
[INT8 FC2](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L3039),
[VAE tiling](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_video_vae.c#L666),
[sampler synchronization](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit.c#L2730).

An independent [native grouped-QKV comparison](https://github.com/9Tempest/m5-max-kernels/blob/7c800fa/docs/grouped-qkv-native.md)
([PR #6](https://github.com/9Tempest/m5-max-kernels/pull/6)) already tested full
Q/K/V projection, Q/K normalization/RoPE and final layouts against this upstream
native dispatch. With two seeds, 25 warmups and 40 repetitions, its custom
candidate was **5.1–5.4% slower at 9,914 rows** and **5.7–7.5% slower at 4,096**,
including online activation packing. That experiment uses synthetic operands
and does not measure this denoiser, but it is a reason to retain native QKV at
these shapes. Earlier single-Q/K fusion gains against a different staged
baseline cannot be treated as gains available over upstream's existing fusion.
PR #6 was subsequently closed unmerged; the slower candidate is not included
as a backend in this H3 fork.

First obtain device attribution before prioritizing QKV versus SDPA versus MLP.
Metal System Trace can reveal queue/CPU/GPU gaps, but unlabeled command intervals
may not carry shader names. GPU Capture/replay and a verified mapping from
fused graph/shader groups to operations are needed for a defensible ranking.
The [optional diagnostic patch](../../../tools/perf/diagnostics.md) records real
geometry, branch choices and sampled post-Euler finite statistics. Its host API
timers add overhead and are not GPU operation timers. Keep diagnostic runs
separate and validate same-seed output parity before interpreting them.

Hadamard INT8 full-model quality/speed, FP8, explicit BF16 splitting, VAE tile
tuning and cached-session acceleration are **not measured in this report**.
An [opt-in native FC2 experiment](../../../tools/perf/experimental-hadamard.md)
includes synthetic correctness and [real-activation follow-up evidence](followup.md).
Neither isolated projection evidence promotes a gain into whole-model speed
or quality. The original paired [summary](sanitized_summary.json) is retained
as its initial evidence snapshot; new measurements have separate summaries.

## Reproduce and extend

Follow [tools/perf/README.md](../../../tools/perf/README.md) for environment setup,
pinned runtime build, model download, two run commands, strict evaluation and
progress analysis. All results require new directories; existing raw JSON and
logs are never overwritten. The standalone CPU tests need no model or GPU.
The shared local GPU lock coordinates these tools with the original kernel
experiment harness; unrelated applications still need to be kept idle manually.

Before claiming a stable improvement, run an independent rerun and a second seed
on identical hardware, interleave variants, collect synchronized warm samples,
include transformation/quantization/padding costs, and define numerical and
quality acceptance criteria before comparing candidate results.
