# Native H256, prepared loading and gate epilogue evidence

The later [QKV/AdaLN-H256 integration study](kernel-integration.md) reports
the new actual-runtime candidate and its negative adoption result separately.

This follow-up keeps the [original one-shot BF16/INT8 observation](README.md)
unchanged. Numerical captures, prepared-checkpoint replay, resident requests
and operator timing are separate experiments. Their elapsed times must not be
combined into a single speedup claim.

The upstream runtime remains `antirez/h3.c` at
`8974cc055ea9c02fcd14cc27dfda3e1027c05153`, with explicitly pinned opt-in
patches. The original BF16 checkpoint is retained. No weights are redistributed.
The [implementation and evidence review](https://github.com/9Tempest/m5-max-kernels/pull/7)
was merged as `c102992b0ce859c728e4df9cf365121fe064e03d` in the originating
benchmark repository. It is separate from upstream h3.c; these experiments
do not require the closed grouped-QKV candidate.

## Five-second fox: same-seed numerical drift

The six BF16/native-INT8/H256 videos use seeds 42 and 43, the same fox prompt,
512×512 resolution and 20 steps. Diagnostic runs capture all 20 packed Euler
states plus final unpacked video/audio VAE inputs. Separate five-second media
exports contain 120 frames at 24 fps and 32 kHz stereo audio.

| Seed | Candidate | Final video latent SNR dB | Final audio latent SNR dB | RGB SNR dB | RGB SSIM | PCM SNR dB |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Native INT8 | 8.5720 | 18.1491 | 12.7128 | 0.690501 | 13.9618 |
| 42 | INT8 + H256 FC2 | 9.4786 | 18.7734 | 13.2407 | 0.704100 | 15.2909 |
| 43 | Native INT8 | 13.0824 | 24.1199 | 17.3740 | 0.833956 | 11.8119 |
| 43 | INT8 + H256 FC2 | 13.8413 | 28.9984 | 19.6092 | 0.848317 | 13.5560 |

H256 is closer to the pinned BF16 path on these final metrics for both seeds.
This is numerical similarity, not perceptual superiority or official-pipeline
parity. Video trajectory relative L2 improves at 18/20 and 17/20 steps; audio
at 12/20 and 19/20. The [real FC2 calibration](followup.md) retains the sampled
tail regression and historical strict-gate failure. The current adoption
policy considers aggregate benefit, tails and complete output together;
implementation tolerances are unchanged. LPIPS, FVD and VBench are unmeasured.

[Playable samples, exact media/evaluation JSON and source hashes](https://github.com/9Tempest/m5-max-kernels/tree/29b2b00d8a80a77e5daa8805111322f93974a7d3/results/published/h3-native-eval/fox-two-seeds-v1).
These capture runs are not latency samples.

## Prepared checkpoint: offline preparation and replay

The derived checkpoint stores all 200 DiT projection payloads. QKV, attention
output and FC1 keep native INT8; FC2 uses H256 whole-row INT8. Other tensors
keep their original precision. The original checkpoint remains separately
available. The derived manifest SHA256 is
`6840c91f6e5b9fcb4e8fe25292a1b1e2209d2f7f9c02e3427cfbbcbc71da4cda`.

One seed42 replay loads 200 prepared projections with zero runtime weight
rotation/requantization. All 42 latent files and both MP4 exports match the
on-load H256 reference byte-for-byte. Verified payload loading takes
12.721376 seconds in this diagnostic run; this is not a repeated load-time or
end-to-end acceleration comparison.

Offline projection preparation takes 7.366674 seconds, separately from
weight loading (1.365344), original tensor hashing (45.582728), and
readback/hash/write (12.995065). Full-model preflight, verification and derived
tree assembly are additional operations. Preparation alone is not the total
cost of creating or verifying the persistent checkpoint.

[Immutable preparation and replay evidence](https://github.com/9Tempest/m5-max-kernels/blob/29b2b00d8a80a77e5daa8805111322f93974a7d3/results/published/h3-native-eval/prepared-checkpoint-summary-v1.json).

## Resident H256: one mode, one seed

After one complete warmup, three same-seed requests report synchronized host
wall intervals:

| Interval | Raw seconds | Median seconds |
|---|---|---:|
| Cached complete generation, including decode/export | 153.602430 / 153.325268 / 153.235321 | 153.325268 |
| Fresh denoise callback | 123.723213 / 123.509807 / 123.386680 | 123.509807 |

All four original 124-frame videos are byte-identical. Audio VAE setup and
compilation APIs repeat per request, so complete-request timing is not
compilation-free. The fresh denoise callback includes synchronized completion
and output copy/unpack. Native cached DiT profile wall spans time since the
previous denoise mark and includes intervening decode; it is not used as the
request's denoise latency. This one-mode session does not establish a stable
cross-mode speedup.

[Original event/timing evidence and preserved harness](https://github.com/9Tempest/m5-max-kernels/tree/29b2b00d8a80a77e5daa8805111322f93974a7d3/results/published/h3-native-eval/performance).

## Native gate epilogues: operator checks before model promotion

The native backend already fuses gate/AdaLN/quantization. Two default-off flags
move gate/residual into the preceding FC2 or attention-output GEMM epilogue,
preserving BF16 branch/residual boundaries and the following AdaLN/INT8 path.
The online chain still has three dispatches: activation packing/rotation,
GEMM and next AdaLN/packing. CPU residual restoration is outside timing.

The first campaign uses real block-0 weights, synthetic inputs with zeros and
outliers, logical rows 2049/9914 with padding 2176/9984, and seeds 17/29. All
twelve cases pass independent full-K selected-row checks, full-output finite
checks, and exact paired hidden/code/scale comparisons. Three warmups and ten
alternating-order repeats per variant include online preparation and padding.

| M9914 policy | Seed | Native median ms | Fused median ms | Native / fused |
|---|---:|---:|---:|---:|
| H256 FC2 | 17 | 14.9160 | 14.9080 | 1.00054 |
| H256 FC2 | 29 | 15.3790 | 15.4130 | 0.99779 |
| Attention output | 17 | 8.3460 | 8.1570 | 1.02317 |
| Attention output | 29 | 8.0410 | 7.5375 | 1.06680 |
| Group1024 FC2 | 17 | 18.1370 | 18.0325 | 1.00580 |
| Group1024 FC2 | 29 | 17.4060 | 16.9490 | 1.02696 |

H256 FC2 has no demonstrated useful gain. The [raw first-campaign package](https://github.com/9Tempest/m5-max-kernels/tree/29b2b00d8a80a77e5daa8805111322f93974a7d3/results/published/h3-native-eval/fused-campaign-001-v1)
preserves all samples and separate CPU encode, submit/wait and supplemental
command-buffer GPU intervals.

The independent second campaign also passes all twelve cases. At M9914,
attention output's native/fused ratio is 1.00788–1.06680 over both seeds and
both campaigns; every observed median favors fusion. M2049 includes a reversal.
H256 FC2 remains inconsistent: M9914 seed17 is only about 0.05%/0.23% faster,
while seed29 is slower in both campaigns. Group1024 FC2 has positive directions
at both shapes but variable magnitude; it is not the H256 policy. These are
descriptive operator medians, not significance tests or model throughput.
Seed42 attention-only and both-flags whole-model captures now pass exact parity
for all 42 Euler/final latent files and both original/exact-export MP4 files
against the on-load H256 reference. Files are re-read for size/hash/finite
validation and runtime source/flag identities are checked. These are diagnostic
results for one request, not performance or every-prompt coverage. Prepared
checkpoint loading is a different runtime path and is not combined with these
fusion flags in this evidence. The completed matched resident comparison below
leaves FC2 off in the primary performance comparison.
The [second raw package](https://github.com/9Tempest/m5-max-kernels/tree/137f2c738bbf7b77208092520dc6f70910c6b583/results/published/h3-native-eval/fused-campaign-002-v1)
and [two-campaign comparison](https://github.com/9Tempest/m5-max-kernels/tree/137f2c738bbf7b77208092520dc6f70910c6b583/results/published/h3-native-eval/fused-comparison-001-002-v1)
preserve every paired sample. The [whole-model parity package](https://github.com/9Tempest/m5-max-kernels/tree/f1e91be5b6bd890ea05309a5ab227b033f827de7/results/published/h3-native-eval/fused-model-parity-seed42-v1)
binds all three immutable capture records and exact output checks.

## Matched resident attention fusion: measured, still descriptive

Both arms use the same frozen fused source, original checkpoint and H256 FC2
policy. Only attention-output gate fusion changes. Each process performs one
complete warmup plus three same-seed timed requests; baseline runs first,
attention second. All eight original 124-frame videos are byte-identical to
one another and the seed42 H256 reference. The comparator validates all 13
runtime harness pins, actual flag/cache markers, event/profile logs, nonoverlap
and media hashes before publishing.

| Synchronized host interval | None raw seconds | Attention raw seconds | Median ratio |
|---|---|---|---:|
| Cached generation | 151.126785 / 151.210439 / 149.808414 | 147.752793 / 149.613678 / 147.617907 | 1.022835× |
| Fresh denoise callback | 121.909995 / 121.987383 / 120.727940 | 118.566479 / 120.543138 / 118.564946 | 1.028200× |

Median generation decreases 151.126785→147.752793 s, by 3.373992 s (2.232557%).
Denoise decreases 121.909995→118.566479 s, by 3.343516 s (2.742610%). These are
ratios of separate medians, not paired repetition ratios or confidence
intervals. One seed and fixed process order do not establish statistical
significance, independent replication or a new best result. Another seed,
reversed process order and independent repeats remain necessary before
stable promotion. Both flags stay off by default.

Generation includes decode and original AV export; repeated audio VAE setup
prevents a compilation-free end-to-end claim. The denoise interval includes
online packing/quantization, padding, synchronized completion, output unpacking
and callback overhead. No latent readback is enabled in these resident runs.
The cached upstream profile wall is not used. Shared-lock scheduling proves
these two arms do not overlap, not that unrelated GPU activity is absent.

[Exact raw timing arrays, source identities and sanitized evidence](https://github.com/9Tempest/m5-max-kernels/tree/f1e91be5b6bd890ea05309a5ab227b033f827de7/results/published/h3-native-eval/fused-resident-seed42-v1).
Prepared loading plus fusion remains unmeasured.

## Fifteen-second first-frame pair

The paired BF16 and prepared-H256 runs use the same full 2,233-byte Chinese
prompt, first-frame image hash and seed83, with a 512×288 canvas, 20 steps and
50 DiT blocks. The entire prompt roundtrips through the pinned native tokenizer;
it is not shortened. The original image is not redistributed. Native output
contains 362 aligned frames; the separate 15-second export contains exactly
360 frames at 24 fps, with 480,000 stereo PCM sample positions at 32 kHz.

Their single diagnostic
process intervals are 506.275859 versus 364.855037 seconds (observed ratio
1.387608, 27.93355% lower). DiT denoising is 443.527 versus 293.568 seconds;
video VAE decoding is 43.918 versus 43.951 seconds. Readbacks are enabled, and
the loaded weights/policies differ, so this is neither a resident comparison
nor evidence of a fused-kernel speedup. DiT tracked tensor peak is
38.199/20.967 GiB; this excludes other process allocations and MPS internals.

All 42 latent tensors per run are finite. Final video shape is
`[24,107,18,32]` and audio shape is `[32,2,603]`; target captures exclude the
persistent first-frame condition rows. Relative to BF16:

| Quantity | Prepared H256 result |
|---|---:|
| Final video latent relative L2 / SNR | 0.21195048 / 13.47531 dB |
| Final audio latent relative L2 / SNR | 0.31593459 / 10.00806 dB |
| RGB SNR / PSNR | 15.69317 / 19.89975 dB |
| Planar-RGB SSIM | 0.81870478 |
| PCM relative L2 / SNR | 0.95068719 / 0.439247 dB |

The waveform drift is substantial. These numerical metrics do not establish
motion quality, prompt adherence or listening quality. This is one seed and
one image-conditioned request; no perceptual pass is assigned.

[Videos, input/recipe manifest and immutable evaluations](https://github.com/9Tempest/m5-max-kernels/tree/137f2c738bbf7b77208092520dc6f70910c6b583/results/published/h3-native-eval/whale-maid-15s-v1),
[separate timing/provenance summary](https://github.com/9Tempest/m5-max-kernels/blob/137f2c738bbf7b77208092520dc6f70910c6b583/results/published/h3-native-eval/performance/whale-maid-15s-diagnostic-v1.json).

## Three additional five-second scenes

The fixed catalog adds person-turn (seed71), rainy-street (seed73), and
water-pour (seed79). Each compares the same prompt/seed at 512×512, 24 fps,
20 Euler passes and all 50 blocks. BF16 uses original weights; H256 uses the
verified prepared checkpoint. Each side has 42 finite captures and a separate
120-frame export with exactly 160000 stereo audio sample positions. These are
different requests, not independent repetitions of the fox performance test.

| BF16-relative diagnostic | Person turn | Rainy street | Water pour |
|---|---:|---:|---:|
| Final video latent relative L2 | 0.09077403 | 0.30224053 | 0.39418745 |
| Final audio latent relative L2 | 0.03419717 | 0.09127944 | 0.18162600 |
| RGB SNR dB | 20.663847 | 6.314006 | 11.772501 |
| Planar-RGB SSIM | 0.91222101 | 0.63843918 | 0.71894218 |
| PCM SNR dB | 12.220583 | 7.021580 | 3.708196 |

All strict evaluations complete, with `quality_decision=not_evaluated`.
Six-frame inspection shows broad walk/turn/wave stages and pouring/level-rise
stages in both modes, with timing and appearance differences. In the rainy
scene, neither sample clearly shows the requested left-to-right car crossing.
Still frames do not establish continuous motion, fluid correctness or human
preference. Audio has not been listened to. The water and fifteen-second
waveform drift especially warrants listening and synchronization review;
numerical similarity is not a perceptual pass.

[Six playable samples and immutable paired evaluations](https://github.com/9Tempest/m5-max-kernels/tree/f1e91be5b6bd890ea05309a5ab227b033f827de7/results/published/h3-native-eval/diversity-three-scenes-v1).

## Cross-review fixes and historical limits

The original-model inventory omitted video/audio VAE config files. Future
runners now verify both against the pinned manifest, including same-size edits
and missing-file rejection. The long first-frame runner separately binds both
configs before its encoder path. The current original and derived files match
the pin, but the [retrospective config audit](https://github.com/9Tempest/m5-max-kernels/blob/f1e91be5b6bd890ea05309a5ab227b033f827de7/results/published/h3-native-eval/performance/runtime-config-audit-15s-v1.json)
is not proof that historical BF16 files stayed unchanged throughout generation.
Historical raw runs are preserved rather than relabeled as runtime-validated.
The matched resident pair ran after the shared inventory fix.

Strict media evaluation now requires the sum of decoded audio-frame samples
to match exact declared duration. Normal decoder skip/discard handling is
already reflected in decoded sample counts; unresolved skip side data or
missing/extra tail samples are rejected. The [audio audit](https://github.com/9Tempest/m5-max-kernels/blob/f1e91be5b6bd890ea05309a5ab227b033f827de7/results/published/h3-native-eval/performance/audio-duration-audit-v1.json)
checks saved, hash-bound fox/15s decoded-frame reports without re-decoding or
recomputing old metrics. Their 160000/480000 sample counts pass. New three-scene
evaluations use the corrected evaluator. Pair evaluation additionally binds
actual video bytes and fixed geometry to each immutable generation record.

Publisher cross-review also tightened original-model association across
prepared runs, exact copied-artifact hashes, private-path rejection and UTC
interval validation. CPU regressions retain the formerly accepted invalid
cases. No original evaluation or generated media was silently rewritten.

## Reproduction and next decisions

The [portable tools](../../../tools/perf/README.md) retain separate clean
baseline, H256, prepared-loader and fusion source pins. They include the latest
config/audio gates and CPU regressions. Directory/import adaptation changes
harness hashes; historical measurements retain their original hashes. A new
report commit in this fork is never accepted as the clean upstream runtime.
Shared GPU locking remains mandatory. No weights or raw latent arrays ship.

The immediate performance candidate is attention-output gate fusion, subject
to another seed and reversed-order repeated resident comparison. H256 FC2 gate
fusion has no consistent operator gain and stays off. Prepared checkpoints
move weight work offline but need their own repeated deployment timing; they
have not been combined with fusion. Video VAE remains around 30 s in the
five-second requests and 44 s at fifteen seconds, making decoder-specific
profiling useful. The partial GPU trace cannot rank QKV, SDPA or MLP kernels;
operator attribution still requires an appropriate device capture. Numerical
and perceptual acceptance remain distinct from timing.

[Machine-readable follow-up evidence and original-source digests](followup-evidence.json).
