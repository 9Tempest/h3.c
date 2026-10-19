# Grouped QKV and AdaLN/H256 integration: negative adoption result

The custom grouped QKV/RMSNorm/RoPE kernel and fused gated
AdaLN/H256/INT8 preparation now execute at the actual H3 QKV/FC1 call sites.
Both remain opt-in. The native grouped fusion is faster in the repeated
operator comparison, and adding QKV/FC1 H256 increases the complete model's
BF16-relative numerical drift in both measured seeds.

The implementation and full evidence are maintained in the
[kernel repository at f4583dbc6a5a106216a9764ff4bd9f6be030863c](https://github.com/9Tempest/m5-max-kernels/tree/f4583dbc6a5a106216a9764ff4bd9f6be030863c/integrations/h3).
The complete MIT-attributed patch applies once to upstream
`8974cc055ea9c02fcd14cc27dfda3e1027c05153`. Do not stack the historical
FC2/prepared/gate patches onto it. This report checkout is not the clean
runtime expected by the pinned source guard.

| Policy | Custom grouped QKV | QKV H256 | FC1 H256 | Existing FC2 H256 |
|---|---:|---:|---:|---:|
| baseline | 0 | 0 | 0 | 1 |
| qkv-only | 1 | 0 | 0 | 1 |
| adaln-h-only | 0 | 1 | 1 | 1 |
| combined | 1 | 1 | 1 | 1 |

The two older gate-in-GEMM flags are zero throughout. The original BF16
checkpoint and prepared FC2-H256 checkpoint are retained. The new policies
prepare weights on load from the original checkpoint; the older derived
checkpoint is not relabeled as QKV/FC1-H256.

## Matched resident performance

| Seed | Synchronized interval | Baseline median, s | Combined median, s | Baseline / combined | Latency change |
|---:|---|---:|---:|---:|---:|
| 42 | Complete cached generation | 147.659017 | 149.364160 | 0.988584× | +1.155% |
| 42 | Denoise callback | 118.525176 | 120.390033 | 0.984510× | +1.573% |
| 43 | Complete cached generation | 147.807164 | 149.768661 | 0.986903× | +1.327% |
| 43 | Denoise callback | 118.640971 | 120.575822 | 0.983953× | +1.631% |

Positive latency change means slower than the existing FC2-H256 INT8 baseline.

| Seed | Interval | Baseline raw seconds | Combined raw seconds |
|---:|---|---|---|
| 42 | Complete cached generation | 149.216220 / 147.646479 / 147.659017 | 149.428955 / 149.364160 / 149.317661 |
| 42 | Denoise callback | 120.058543 / 118.455572 / 118.525176 | 120.443899 / 120.390033 / 120.381656 |
| 43 | Complete cached generation | 147.824244 / 147.807164 / 147.701214 | 149.983710 / 149.768661 / 149.755262 |
| 43 | Denoise callback | 118.640971 / 118.655992 / 118.593650 | 120.794514 / 120.575822 / 120.554058 |

Each of the four processes uses one full warmup and three timed requests.
Order is baseline42, combined42, combined43, baseline43. The recipe is the
same fox prompt, 512 by 512, five seconds, 20 steps and 50 blocks, with
reference RoPE and no reuse/token reduction. No activation/latent readbacks
run during timing. All four original exports in each arm are byte-identical
and match the corresponding diagnostic capture's original video.

Generation includes decoding and original 124-frame AV export, but excludes
preflight/build, initial weight preparation and the separate 120-frame trim.
Repeated audio-VAE setup prevents a compilation-free end-to-end claim.
Fresh synchronized denoise callbacks include online activation preparation.
These are descriptive medians with two seeds, not confidence intervals,
perceptual certification or a new fifteen-second throughput claim.

## Numerical and operator evidence

Two reversed operator campaigns pass 32 supported QKV cases and 48 preparation
cases. Eight QKV M17 records remain explicitly unsupported; M129 supplies
supported tail coverage. Preparation's fused/staged BF16 values, INT8 codes
and FP32 scales are exact. Independent implementation and quantization errors
remain separately reported. No tolerance was relaxed.

At M9914/17455, the custom QKV activation-pack-plus-kernel path is 6.1–6.9%
slower than native across both seeds and rounds. Fused H256 preparation beats
staged H256 in the measured gate/shared-map cases, but adding H256 costs more
than native preparation without it. Earlier single-Q/K microbenchmark wins
therefore do not transfer to the already-fused grouped native comparator.

All four model policies have finite 20-step/final captures and decoded media
evaluations at seeds 42 and 43. Disabled new flags exactly reproduce the
historical FC2-H256 baseline's 42 latent tensors and both MP4 exports.

| Seed | Policy | Final video L2 | Final audio L2 | RGB SSIM | PCM SNR, dB |
|---:|---|---:|---:|---:|---:|
|42|baseline|0.335791|0.115167|0.704100|15.2909|
|42|combined|0.403528|0.197490|0.651836|10.9949|
|43|baseline|0.203205|0.035488|0.848317|13.5560|
|43|combined|0.373968|0.065682|0.732474|11.1246|

All values are relative to the same-seed BF16 capture. Combined video L2
regresses at 14/20 and 19/20 Euler states; audio at 18/20 and 16/20. Audio
absolute-error p999 also grows. In contrast, all 576 sampled rows in the
combined real-activation calibration have lower same-X/W projection L2 with
H256. Three maximum-error regressions are retained. The repeatable complete
model regression drives rejection; a single row is not an automatic veto.
QKV-only improves final video agreement but worsens final audio latent error
and is slower in the operator comparison. It also remains experimental.

Six sampled frames per candidate retain the fox/snow scene with pose/timing
changes. Continuous motion and audio listening remain unassessed. SNR/SSIM
do not certify prompt adherence, perceptual quality or official-Python parity.

## Reproduction, samples and next experiments

- [Complete integration patch, contracts and commands](https://github.com/9Tempest/m5-max-kernels/blob/f4583dbc6a5a106216a9764ff4bd9f6be030863c/integrations/h3/kernel-integration.md)
- [Raw JSON, source snapshots, six new videos and authored results](https://github.com/9Tempest/m5-max-kernels/tree/f4583dbc6a5a106216a9764ff4bd9f6be030863c/results/published/h3-native-eval/kernel-integration-v1)
- [Machine-readable evidence links](kernel-integration-evidence.json)

Next experiments should match the native 128-row/full-K/local-scale QKV layout
before isolating tile/traversal choices; separate QKV and FC1 rotation before
layer/step calibration; and reduce H256 preparation traffic/reductions while
retaining BF16 boundaries and residual output. Source differences suggest
ablations, not measured hardware-bottleneck attribution. Every variant still
needs independent references and whole-model aggregate/tail/media review.

The source pin is
`0836659da9c9ea35fd6ab9956cfbc88d0fd2063866848487d40a6885f87e89d5`,
the complete source digest is
`7b694d2313a13012dab2714f15d1b2a58adff4b067230857dae284f0744c12d0`,
and the complete patch digest is
`f92f680a19b1ab81ab3eccead7c70720ccaa7a41c729da61fdc774d7eff6c01c`.
No weights, raw latent arrays or user reference image are redistributed.
