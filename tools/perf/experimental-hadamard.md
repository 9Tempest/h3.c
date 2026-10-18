# Native H3 FC2 Hadamard experiment

The optional native implementation rotates **FC2 only**, after SwiGLU. It does not implement a fully Hadamard-rotated H3 model. The original checkpoint remains the numerical reference and is retained locally. No checkpoint replacement is authorized by the results below.

Upstream is [`antirez/h3.c` at `8974cc055ea9c02fcd14cc27dfda3e1027c05153`](https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153). Model files are [`MiniMaxAI/MiniMax-H3` at `42ed227ee7df40d41602854ae760620d6eb651fe`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe). The runtime's MIT license and model's separate license still apply; the repository does not distribute weights.

## Precision contract

For every active DiT block, the experimental FC2 path uses the following operations:

1. At model loading, rotate each original BF16 FC2 weight row using the normalized, plain Sylvester H256 transform. Butterflies and normalization use FP32, without signs or a transformed BF16 rounding boundary. Quantize each complete rotated row to symmetric INT8 with a single FP32 maximum-absolute-value scale.
2. At inference, read the existing **BF16 SwiGLU output**, apply the same FP32 H256 transform, then quantize each complete activation row. A single Metal dispatch implements this preparation through SIMD shuffles and register butterflies, without Python/MLX transfers or a temporary FP32 device tensor.
3. Reuse the native FC2 INT8 TensorOps projection: INT32 dot products, FP32 scales, then BF16 projection output. Continue with the original residual gate and normalization.

FC1, post-AdaLN BF16 rounding, SwiGLU BF16 storage, residual updates, and Q/K normalization/RoPE are unchanged. Rotations do not move through SwiGLU. The path supports the actual FC2 width K=14336 and preserves zero padding to 128-row tiles. Zero H256 rows use codes 0 and scale 1. The native quantizer uses scale 1/127 for genuine zero rows and scale 1 for padded rows; decoded zeros are identical.

The baseline is the deployed native **activation group1024** FC2, with 14 activation scales per row and one scale per output weight row. An additional native plain-row baseline is retained as an ablation. H256 whole-row must outperform group1024 to be considered for adoption; beating plain-row alone is insufficient. A future H256+group1024 policy would require a distinct mode, corresponding reference, and measured evidence.

`H3_HADAMARD_FC2=1` opts in. The default is unchanged. Unsupported TensorOps, invalid K, conflicting row/stage/clipping overrides, and compilation failures are explicit failures. No fallback is promoted as an experiment result. This version does not persist rotated weights: their preparation is included in model loading.

## Pinned implementation and build

The integration uses a separate checkout, preserving the original deployment. The patch changes `h3_dit.c`, `h3_gpu.h`, `h3_gpu.m`, and `h3_shaders.metal`.

| Artifact | SHA256 |
|---|---|
| [hadamard.patch](hadamard.patch) | `84065e6139c842aee2cb6d037a6c72ecfa9895e07b1696e9bf638bde3a5a8d1e` |
| Tracked source digest, using `run.git_provenance` | `833076975eeaaf14b0a9db01f6237d7de77930034222b1dd3b85cca7721f04ec` |

From this fork root:

```sh
git clone https://github.com/antirez/h3.c .cache/h3-hadamard
git -C .cache/h3-hadamard switch -c codex/h3-hadamard 8974cc055ea9c02fcd14cc27dfda3e1027c05153
git -C .cache/h3-hadamard apply ../hadamard.patch
make -C .cache/h3-hadamard -j8
uv sync --locked
```

Use a new path if the checkout already exists. The standalone runner validates the complete patched source digest and patch digest, rejects untracked source additions, then rebuilds the executable from the pinned source. It also verifies every required model weight against the pinned manifest. This reads the weights before timing and warms the filesystem cache.

Build the independent projection readback driver:

```sh
clang -O2 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Werror \
  -I .cache/h3-hadamard tools/perf/fc2_validate.c \
  .cache/h3-hadamard/libh3.a -framework Foundation -framework Metal \
  -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
  -framework Accelerate -licucore -lm -o .build/h3-fc2-validate
```

The model component directory for the driver is **`FL2VA/transformer`**, and its weight key is `blocks.<block>.mlp.fc2.weight`.

## Full generation and diagnostic capture

[run_fc2_experiment.py](run_fc2_experiment.py) fixes the fox-in-snow prompt, 512×512 canvas, 20 Euler passes, all 50 blocks, seed, reference RoPE, and reuse/core-reuse of 1. Five requested seconds align to 124 native frames at 24 fps; the original is retained and a separate 120-frame/5-second video is exported and verified. The GPU lock covers source/model preflight, compilation, the entire process, capture checks, and media validation.

Use new immutable output directories for every invocation:

```sh
uv run python tools/perf/run_fc2_experiment.py \
  --mode upstream-int8 --seed 42 --capture-fc2 --capture-latents \
  --output-dir results/local/h3/fc2-native-capture-seed42-NEW

uv run python tools/perf/run_fc2_experiment.py \
  --mode bf16 --seed 42 --capture-latents \
  --output-dir results/local/h3/fc2-bf16-capture-seed42-NEW

uv run python tools/perf/run_fc2_experiment.py \
  --mode hadamard-fc2 --seed 42 --capture-latents \
  --output-dir results/local/h3/fc2-hadamard-capture-seed42-NEW
```

`--gpu-wait-seconds 900` permits waiting for another experiment. The three BF16 projection overrides are applied automatically in BF16 mode. Do not inherit arbitrary `H3_*` variables; the runner rejects them. `--capture-fc2` is restricted to an INT8 FC1 trajectory and is incompatible with BF16 mode.

FC2 captures sample steps 0/10/19 and blocks 0/24/49. Each record contains complete K=14336 rows: each packed modality's first/middle/last row, plus seven uniformly spread rows, with duplicates removed. Metadata records row IDs and modality boundaries. These are BF16-stored post-SwiGLU activations on the native INT8 FC1 trajectory, not activations from a fully BF16 model. Capturing adds nine explicit submit/wait/restart points.

Latent capture records every completed Euler step in packed FP32 model coordinates and the final unpacked VAE inputs. For this recipe, final video is `[24,37,32,32]` and final audio is `[32,2,207]`. It uses existing completed GPU windows, adding CPU readback/I/O. Metadata contains shapes, precision modes, completed step count, final sigma, timestamps, and the caller-declared seed, which the runner validates against its request manifest. Every binary is checked for expected size and finite values. All files are written exclusively.

**Capture runs are diagnostic and cannot serve as latency samples.** A performance invocation omits both capture flags. Its timing still includes shader setup, loading, weight preparation, text encoding, denoising, VAE, and export; it is not compilation-free resident or kernel-only latency. Build and full-weight hash preflight are recorded separately.

## Independent FC2 references

[fc2_validate.c](fc2_validate.c) calls the same native FC2 dispatcher as full inference. It accepts the original component directory, block number, frozen input BF16 file, a new empty output directory, logical M, and shader path. It evaluates native group1024, H256 whole-row, and native plain-row against the **same frozen X/W**. Run the driver only while holding `locking.gpu_lock()` for the complete child process; the full-generation runner does not hold a lock after it returns.

The driver computes all 5376 output channels. It dumps all output values and activation codes/scales, plus 32 uniformly selected weight/output channels including both boundaries. Full K is retained for each reference. When logical M exceeds captured rows, the remaining input rows are zero; synthetic validation additionally exercises nonzero last rows of partial tiles.

[fc2_analyze.py](fc2_analyze.py) runs on CPU:

```sh
uv run python tools/perf/fc2_analyze.py NEW_READBACK_DIRECTORY \
  --output NEW_ANALYSIS.json
```

The analyzer separates:

- **Implementation error:** GPU output versus independent NumPy float64 products of the actual decoded INT8 operands. BF16-output relative-L2 gate is 0.008.
- **Quantization error:** decoded products versus float64 products of the original BF16 X/W.
- **Total error:** actual BF16 GPU output versus the original-operand reference.

It independently checks normalized H256 dot preservation, maximum-absolute-value scales, quantization cells, finite outputs, exact zeros, and padded rows. Exact CPU-reference code mismatches are also reported; a predefined 0.0001 quantization-cell allowance covers FP32 divide/reciprocal rounding. Every output is scanned for finite values, but independent projection error covers **32 selected output channels**, not every channel or the full model. The analyzer records its own source hash and all readback hashes.

## Promotion gate and evidence

Before looking at real-activation results, the numerical gate is fixed: for **every** sampled step×block, H256 must strictly improve aggregate relative L2, RMSE, and maximum absolute error against native group1024, with no sampled-row L2 regression. All implementation/packing checks must pass. Per-cell worse counts remain visible; an average improvement cannot conceal a deteriorating layer. No tolerance is loosened after seeing results.

Projection sampling is necessary but insufficient. Promotion additionally requires independent seeds, same-seed complete denoiser trajectories and final video/audio latents against BF16, and no visual/audio quality regression. Final comparisons use exported FP32 latents rather than MP4 pixels to avoid conflating model error with VAE/media encoding. Until all criteria pass, retain the original checkpoint. The analyzer always reports `checkpoint_replacement_eligible=false`; no script deletes or replaces weights.

The first GPU correctness campaign passed all three policies on actual block-0 FC2 weights with synthetic BF16 inputs: seeds 17/29, M=129/257, explicit zero rows, outliers, and nonzero final logical rows. [Sanitized synthetic correctness evidence](../../docs/performance/m5-max-512-5s/fc2-synthetic-summary.json). Implementation relative L2 was about 0.0016–0.0017, within the unchanged BF16-output gate. H256 total error was 1.35% for each seed, versus native group1024's 1.75% and 1.86%. These are synthetic distribution checks and do **not** satisfy the real-activation or model-quality promotion gate. Real trajectory/capture analysis is recorded separately as it completes.

Host C compilation, offline Metal 4 shader compilation, and 13 CPU reference/runner tests passed before the generation capture. Source compilation alone was not counted as GPU correctness.
