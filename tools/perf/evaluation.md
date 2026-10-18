# H3 evaluation: paired drift and model quality

`tools/perf/evaluate.py` measures numerical differences between matched
outputs. It supports captured activations/latents (`.npy` or matching `.npz`
keys) and decoded video/audio. It is independent of `m5kernels`, `run.py` and
repository layout, so the same file can be copied into the pinned `h3.c` fork.
Requirements are Python 3.11+, NumPy, and FFmpeg/FFprobe for media. It never
downloads evaluation models or runs a GPU backend.

BF16 is the comparison baseline, not perceptual ground truth. Same-prompt,
same-seed generation can diverge spatially after small numerical changes.
Low pixel similarity therefore identifies output drift; it does not by itself
show that the candidate video is worse. A completed metric calculation is not
a quality pass. Every record keeps `quality_decision: not_evaluated`.

## Metric choices and the available evidence

| Metric | Appropriate scope | Interpretation and boundary |
| --- | --- | --- |
| SNR, relative L2, RMSE, maximum absolute error | Matched projection outputs, latents, RGB frames or PCM samples | Numerical drift at the exact capture boundary; sensitive to alignment and signal energy, not semantic quality |
| PSNR | Matched arrays with an explicit data range | Global MSE followed by a logarithm; not the average of frame PSNR values; signal fidelity, not a generative-quality score |
| FFmpeg SSIM | Corresponding decoded frames | Structural similarity under a named implementation/color contract; no temporal or semantic guarantee |
| LPIPS | Corresponding RGB images/frames | Learned image perceptual distance; lower is more similar; frame averages do not measure temporal consistency |
| VBench custom-input dimensions | A suitable custom clip or clip set | Dimension-specific visual/temporal measures; a single clip cannot stand in for the standard prompt-suite aggregate |
| VBench 2.0 | Dimension-specific benchmark inputs and, for some dimensions, collections | Intrinsic realism evaluation; not a generic score applicable to one arbitrary fox clip |
| FVD | Collections of videos from two distributions | Feature-distribution comparison; one clip per variant is insufficient for a meaningful estimate |

The original [SSIM paper and author implementation](https://ece.uwaterloo.ca/~z70wang/research/ssim/)
introduce a structural fidelity measure. The
[FFmpeg PSNR](https://ffmpeg.org/ffmpeg-filters.html#psnr) and
[SSIM filter documentation](https://ffmpeg.org/ffmpeg-filters.html#ssim) describe
their own frame-based implementations. Preserve the implementation, pixel
format, range and frame correspondence; do not interchange FFmpeg SSIM with
another library's SSIM or report RGB SSIM as the historical YUV score.

[LPIPS](https://github.com/richzhang/PerceptualSimilarity), based on
[Zhang et al.](https://arxiv.org/abs/1801.03924), accepts RGB tensors shaped
`N×3×H×W`, normalized to `[-1,1]`. A future adapter must pin the package/revision,
backbone, metric version and checkpoint hash. For aligned frames it should
save every distance and the mean/median/p95/maximum. These remain image-distance
statistics, without a claim about prompt adherence, motion or audio.

The [VBench official repository](https://github.com/Vchitect/VBench) and
[paper](https://arxiv.org/abs/2311.17982) define 16 dimensions. Its documented
custom-input mode supports subject consistency, background consistency,
motion smoothness, dynamic degree, aesthetic quality and imaging quality.
Report those dimensions separately as `custom_input`; do not call a fox-clip
evaluation the complete VBench score. Dynamic degree measures motion and is
not a universal higher-is-better quality target.

[VBench 2.0's official documentation](https://github.com/Vchitect/VBench/tree/master/VBench-2.0)
and [paper](https://arxiv.org/abs/2503.21755) evaluate intrinsic realism. Its
documented custom-input support covers human anatomy/identity/clothes,
diversity and multi-view consistency; diversity requires at least 20 videos
per prompt. The human-specific dimensions are not applicable to this fox
request, and one pair cannot support a full benchmark claim. Pin the selected
dimension and current evaluation protocol when implementing an adapter.

The [original FVD paper, section 4.2](https://arxiv.org/pdf/1812.01717) compares
feature means and covariances and shows finite-sample estimation effects:
even samples from the same distribution generally yield nonzero distance,
and compared experiments must use equal sample counts. The 120 frames of one
clip are not 120 independent video samples. Overlapping windows do not fix
this limitation. [Ge et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ge_On_the_Content_Bias_in_Frechet_Video_Distance_CVPR_2024_paper.html)
also identify I3D-FVD's content bias and limited temporal sensitivity.
[Beyond FVD](https://arxiv.org/html/2410.05203v2) gives dataset-specific
examples of slow sample-size convergence; those results are not a universal
minimum sample count. A future collection study must pin its feature model,
preprocessing and frame sampling, equalize video counts, and report repeated
subsampling/uncertainty alongside the sample count.

## Exact numerical contract

For reference `r`, candidate `c`, error `e = c - r`, and `N` scalar elements:

- `SNR_dB = 10 log10(sum(r²) / sum(e²))`.
- `relative_L2 = sqrt(sum(e²) / sum(r²))`.
- `RMSE = sqrt(sum(e²) / N)`; maximum absolute error is `max(abs(e))`.
- `PSNR_dB = 20 log10(data_range / RMSE)`.

All elements are checked for finiteness and included in the metrics; there is
no sampling, clipping or automatic range estimation. The implementation uses
float64 with scaled norms to avoid overflowing a sum of squares. Subtraction
or conversion overflow rejects the numerical result. Integer inputs outside
the exactly representable float64 range are rejected. BF16 bit patterns must
be decoded to numeric floating arrays by the producer, not disguised as
`uint16` values. Inputs must have the same nonempty shape; `.npz` keys must
match exactly, and pickle/object/complex inputs are rejected.

Arrays require `--data-range`, chosen and recorded before comparing variants.
RGB is decoded as unsigned 8-bit RGB with `data_range=255`. PCM uses floating
sample values without amplitude normalization, with default `data_range=2`
for the full nominal `[-1,1]` span. `--audio-data-range` can explicitly select a
different convention; this is not the peak-amplitude-1 convention in the older
`compare.py`. SNR is independent of this PSNR range choice. PCM outside its
nominal full-scale interval is preserved, not clipped.

Each scalar is `{status, value}`. JSON never contains NaN or Infinity:

| Condition | SNR | Relative L2 | PSNR |
| --- | --- | --- | --- |
| Nonzero reference, zero error | `positive_infinity`, null | finite 0 | `positive_infinity`, null |
| Zero reference, zero error | `undefined`, null | `undefined`, null | `positive_infinity`, null |
| Zero reference, nonzero error | `negative_infinity`, null | `positive_infinity`, null | finite if representable |
| Nonfinite input | `invalid_nonfinite`, null | `invalid_nonfinite`, null | `invalid_nonfinite`, null |

Reporting-range overflow/underflow has an explicit state; it is never replaced
with a large finite sentinel. Numeric failure retains the input hashes and
failure record and makes the command fail.

## Strict media correspondence

The media path requires exactly one video and one already-32-kHz stereo audio
stream. It compares dimensions, nominal/average FPS, exact rational frame
timestamps, audio frame timestamps and sample counts, declared duration,
pixel/color metadata and aspect ratio. Each video must have a contiguous
constant-rate timeline; audio blocks must have no gaps or overlaps and must
start with video. Declared audio/video durations must match. Unsupported
rotations are rejected.

Frame durations must also match the declared FPS/audio sample counts; changing
channel layouts, frame cropping, interlacing and repeated fields are rejected.
Frame-level color metadata is preserved and compared as well as stream metadata.

FFprobe records the decoded frame timeline before evaluation. FFmpeg then
uses software decoding (`-hwaccel none`), no autorotation, passthrough video
timing, `rgb24`, and `pcm_f32le`. There is no scale, frame-rate conversion,
frame duplication, audio rate/channel conversion, shift search or time warp.
Raw byte/sample counts must agree with the probe. Decoder priming/padding
handling is FFmpeg's documented decode behavior; the harness does not trim
the decoded sample arrays to force agreement. Global stereo metrics and
separate left/right metrics are recorded.

Matching unknown color metadata is retained as unknown, not evidence of a
calibrated color space. RGB conversion follows the recorded FFmpeg version
and input metadata. MP4 metrics include codec error and therefore do not
isolate model quantization. Prefer corresponding pre-encode frames or
lossless outputs for a future precision experiment.

`--ssim` compares the same decoded RGB frames, converted to planar RGB `gbrp`
inside the FFmpeg SSIM filter. It retains every frame score, their arithmetic
mean and raw stdout/stderr. This is an explicitly named RGB FFmpeg SSIM
backend. Missing frames, nonfinite SSIM or backend failure produce failure,
with no alternate metric fallback.

## Run and preserve evidence

```sh
uv run python tools/perf/evaluate.py reference-latents.npz candidate-latents.npz \
  --kind arrays --data-range 1 --output-dir results/local/h3/eval-latents-001

uv run python tools/perf/evaluate.py bf16/video-5s.mp4 int8/video-5s.mp4 \
  --kind media --ssim --output-dir results/local/h3/eval-media-001

uv run python tools/perf/evaluate.py bf16/video-5s.mp4 bf16/video-5s.mp4 \
  --kind media --ssim --output-dir results/local/h3/eval-self-001

uv run pytest -q tools/perf/tests/test_h3_evaluate.py
```

In a standalone fork, use `python path/to/evaluate.py` with NumPy installed;
no repository-relative imports are required. Use a fresh output directory.
Each evaluation retains `evaluation.json`, input hashes, source hash/revision
and dirty state when Git is available, library/tool versions, tool binary
hashes, raw probe/process logs, decoded media and artifact hashes. Existing
records cannot be overwritten. Input hashes are checked again at the end.
Evaluation wall time is CPU evaluation cost, not generation latency.

LPIPS and VBench currently have no configured adapter or checkpoint. Without
their flags they are `not_measured`; explicitly requesting `--lpips` or
`--vbench` records `unavailable`, leaves the numeric metrics intact, and makes
the overall result `partial` with a nonzero exit code. There is no download or
silent substitute. FVD is `not_measured` for this single-pair interface.

## Promotion protocol

The first strict evaluation compares the completed seed42 BF16 and upstream
INT8 exports. Both pass geometry, exact timestamp/duration, color-metadata and
audio alignment checks: 120 frames at 512×512/24 fps and 160000 stereo samples
at 32 kHz. All decoded samples are finite.

| Global paired metric | RGB (`data_range=255`) | Stereo PCM (`data_range=2`) |
| --- | ---: | ---: |
| SNR, dB | 12.712753 | 13.961846 |
| Relative L2 | 0.23139946 | 0.20040461 |
| RMSE | 29.858543 | 0.001710915 |
| Maximum absolute error | 233 | 0.020317929 |
| PSNR, dB | 18.629431 | 61.356032 |

Mean **RGB FFmpeg SSIM is 0.690500567**. This differs from the earlier
`compare.py` YUV SSIM (0.806325) because the color/metric contract differs; it
is not a performance or quality regression between repeated evaluations.
BF16 self-comparison gives zero RGB/PCM error and SSIM 1, with positive-infinite
SNR/PSNR represented by status plus null. The published results are in
[paired and self-pair metrics](../../docs/performance/m5-max-512-5s/sanitized_summary.json).
Both were generated by evaluator SHA256
`e54704bba58a6c85bf2bddd6825f7ecbb39b4a50440f6a350708fdd3e5e62f8a`.
The strict evaluator enforces each frame's duration, cropping/field flags,
audio layout and frame-level color metadata. These are drift measurements from one
seed and do not authorize checkpoint promotion.

## Promotion requirements

1. Match source/model hashes, prompt, seed, sampler, steps, dimensions and
   conditioning. Capture identical projection boundaries and early/middle/late
   denoiser states. Separate implementation error against decoded operands
   from total error against the BF16 baseline. Require finite tensors.
2. Run the paired numerical harness and a self-pair sanity check. Review
   projection and latent trajectories, not only final pixels. Record drift
   without inventing a universal SNR/PSNR threshold for model quality.
3. Use a preregistered diverse prompt set and multiple seeds, then independent
   reruns on the same hardware. Inspect visual fidelity, temporal consistency,
   prompt adherence and audio quality/synchronization; use applicable
   perceptual/VBench dimensions and blinded human comparisons. Visual metrics
   do not evaluate stereo audio or audio/video synchronization by themselves.
4. Promote a checkpoint/backend only when the numerical contracts and defined
   video/audio quality gates do not regress and the complete deployment path
   improves. Report startup, preparation, resident inference and export scopes
   separately. Run CPU evaluation outside timed performance windows, and do
   not multiply a projection microbenchmark speedup into model throughput.

The current harness supplies the numeric/structural evidence for this
protocol. It intentionally does not implement a model-quality approval gate
or claim that a single five-second clip validates a checkpoint.
