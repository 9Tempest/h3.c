# Real FC2 activations and a partial denoising trace

The native diagnostic run completed with **9,914 actual packed rows** and
produced a five-second MP4 **byte-identical to the earlier unmodified INT8
sample**. On frozen real FC2 inputs, H256 whole-row INT8 reduced aggregate
relative L2, RMSE and maximum absolute error in **all nine sampled operations**
versus upstream group1024. Of **144 sampled row instances**, **143 improved**
in relative L2 and **one regressed**. This page records the initial calibration
stage; later two-seed output checks and deployment evidence are organized in
the [native experiment report](native-experiments.md).

The initial preregistered rule required no sampled-row L2 regression, so its
historical result is **failed**. After reviewing aggregate improvement and
the tail case, the study owner changed the adoption policy to consider overall
benefit, tail risk and full-model output together. A single row is no longer
an automatic veto. The old rule, all raw metrics and its result remain
unchanged; no tolerance was relaxed to conceal an implementation error.

## What was captured and checked

The opt-in [native patch and tools](../../../tools/perf/experimental-hadamard.md)
retain the pinned upstream source and original checkpoint. The native run used
the same fox prompt, seed 42, 512-square canvas, 20 Euler passes, 50 blocks and
reference RoPE as the [baseline report](README.md). Observed capture segments
are 28 text rows, 414 stereo-audio rows and 9,472 video rows. All nine captured
post-SwiGLU BF16 samples and all recorded Euler/final latents were finite.

The output SHA-256 is
`993e73621f4028c9759058457f0f41ee2a06dd865bf8691de28f06baddcc3020`,
identical to the unmodified INT8 five-second MP4. This establishes output
equivalence for this one seed/configuration, not every layer or every seed.
Capture adds nine explicit GPU waits plus latent readback and I/O; a trace was
also attached. **Its elapsed time is excluded from baseline performance.**
See [native-diagnostics-summary.json](native-diagnostics-summary.json).

Each of steps 0, 10 and 19 and blocks 0, 24 and 49 supplies 16 complete K=14,336
rows, spanning modality boundaries and spread-out positions. The inputs follow
the **native INT8 FC1 trajectory**, not a fully BF16 or H256 trajectory. The
projection driver compares group1024, H256 whole-row and plain-row INT8 using
the same frozen X/W. It embeds the 16 rows in logical M=129 with zeros and pads
to M=256. This is an isolated numerical experiment, not a full M=9,914 kernel
timing or correctness sweep.

All 5,376 produced output channels are scanned for finite values. Independent
NumPy float64 references use the full K and **32 selected output channels**;
they separate packing/implementation error from quantization and total error.
All three policies passed the independent implementation and packing checks
in all nine cases. Implementation relative L2 ranged from **0.001350 to
0.002167**, below the unchanged **0.008** BF16-output limit. This limit is an
implementation check, not a model quality threshold.

## Projection benefit and tail risk

Relative L2 below compares actual BF16 GPU output against the original BF16
operand reference, over the sampled 16 × 32 output values. All nine operations
also improve aggregate RMSE and maximum absolute error with H256.

| Step | Block | Group1024 relative L2 | H256 relative L2 | Row instances improved / regressed |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.016364 | 0.013874 | 16 / 0 |
| 0 | 24 | 0.027985 | 0.011384 | 16 / 0 |
| 0 | 49 | 0.017331 | 0.007984 | 16 / 0 |
| 10 | 0 | 0.017523 | 0.013658 | 16 / 0 |
| 10 | 24 | 0.028230 | 0.013132 | 16 / 0 |
| 10 | 49 | 0.023042 | 0.007780 | 16 / 0 |
| 19 | 0 | 0.018729 | 0.011293 | 16 / 0 |
| 19 | 24 | 0.026001 | 0.012423 | 16 / 0 |
| 19 | 49 | 0.017438 | 0.006201 | 15 / 1 |

The regression is step 19, block 49, sample index 14: packed row **7,434**,
video row **6,992**, token coordinate **(t=27, y=5, x=0)** in the 16 × 16
spatial token grid. Over its 32 independent reference channels:

| Error | Group1024 | H256 |
| --- | ---: | ---: |
| Relative L2 | 0.0071092561 | 0.0081929306 |
| RMSE | 19.589760 | 22.575857 |
| Maximum absolute error | 61.106137 | 107.062750 |

Per-cell worse counts remain in [fc2-real-summary.json](fc2-real-summary.json),
including in operations whose aggregate improves. Those counts and the tail
case must remain visible when judging the benefit. Whole-denoiser trajectory,
second-seed results and video/audio evidence are separate from this projection
experiment; see the [subsequent evidence](native-experiments.md). An isolated
FC2 result does not establish faster inference or a better generated clip.
The original checkpoint remains the reference and is retained.

## Partial Metal System Trace

A requested 15-second attachment during this diagnostic run exported **1,665
compute intervals**. The span from the first to last observed interval was
**10.870592 seconds**; their active union was **10.773801 seconds**, or
**99.11% of that observed span**. This denominator is not the requested full
15 seconds. The artifact retains source/export hashes in
[denoise-trace-summary.json](denoise-trace-summary.json); process identifiers,
raw XML and the large binary trace are not published.

This sample shows sustained recorded device work within that window. It does
not measure shader occupancy, arithmetic throughput, whole-model utilization
or uninstrumented baseline GPU time. The intervals carry no verified mapping
to QKV, SDPA or MLP, so they cannot rank these operations. GPU Capture/replay
or verified operation-level device instrumentation remains the next step for
that attribution. Profiling and capture overhead are explicit.

The independent grouped-QKV candidate was slower than the pinned native path
at large shapes. Its [PR #6](https://github.com/9Tempest/m5-max-kernels/pull/6)
was closed unmerged, and it is not retained as an H3 deployment backend. The
[archived comparison](https://github.com/9Tempest/m5-max-kernels/blob/7c800fa/docs/grouped-qkv-native.md)
remains a useful negative result; these H3 tools do not depend on that candidate.
