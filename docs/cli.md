# CLI Reference

```bash
qlean --model_name MODEL --model_path DIR --save_path DIR [options]
```

## Required Arguments

| Argument | Description | Example |
|:---|:---|:---|
| `--model_name` | Model identifier for built-in recipe matching. Use the official HuggingFace model stub. | `Qwen/Qwen3-235B-A22B` |
| `--model_path` | Local path to the source model directory | `/path/to/Qwen3-235B-A22B` |
| `--save_path` | Output directory | `/path/to/Qwen3-235B-A22B-INT8` |

## Optional Arguments

| Argument | Default | Description |
|:---|:---|:---|
| `--output-format` | `INT8_W8A8` | Target output format |
| `--input-format` | Auto-detect | Source format: `BF16` / `FP8` / `NVFP4` |
| `--recipe` | Auto-select | Path to a custom Recipe YAML |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--verify` | follow Recipe | `off` / `bound`. Overrides `verify` in the Recipe; applies only to `GeneralQuantEngineModifier`. See [verification.md](verification.md). |

## Helper Commands

| Command | Purpose |
|:---|:---|
| `qlean --list-formats` | List all supported conversion paths and aliases |
| `qlean --list-models` | List all models with built-in recipes |
| `qlean --list-models MODEL_NAME` | Check recipe support for a specific model |
| `qlean --examples` | Show usage examples |
| `qlean -h` | Show full help |

## Format Names

The source format is auto-detected from the model files and rarely needs to be set manually. `--input-format` accepts only `BF16`, `FP8`, and `NVFP4`; `FP8` means an FP8 block-wise checkpoint.

Target names for `--output-format` and their accepted aliases. All names are case-insensitive.

| Format | Aliases |
|:---|:---|
| INT8_W8A8 | W8A8, INT8 |
| FP8_BLOCK | FP8-BLOCK |
| FP8_DYNAMIC | FP8-DYNAMIC, FP8-CHANNEL, FP8_CHANNEL |
| MXFP4 | — |
| GPTQ_INT4 | — |
| GPTQ_INT8 | — |
| GPTQ_MXFP4 | — |
| GPTQ_INT4_GROUP | — |
| GPTQ_INT8_GROUP | — |
| BF16 | — |

`GPTQ_INT4` is the only per-channel GPTQ INT4 output name. Automatic INT4 recipes select only routed expert Linear weights under `mlp`, `ffn`, `feed_forward`, or `block_sparse_moe`: symmetric, static, per-channel integer INT4, with no `group_size` or activation quantization. Gates, shared experts, attention, and other unselected layers stay BF16 when loaded in BF16. An explicit recipe is kept unchanged and may additionally select disjoint symmetric per-channel INT8 weights; grouped, asymmetric, or dense INT4 and activation/KV-cache quantization are not supported for this output.

`GPTQ_INT4_GROUP` and `GPTQ_INT8_GROUP` use symmetric integer INT4/INT8 weights in groups of 128 along the input dimension, with one scale per group in the model dtype, matching llmcompressor's `W4A16` / `W8A16` recipes. They save through the standard compressed-tensors `pack-quantized` format (int32-packed weights plus `weight_scale` tensors declared in `config.json`) and are reloadable by standard compressed-tensors/Hugging Face weight-only loaders. Unlike `GPTQ_INT4`, automatic group recipes select all Linear weights, not just routed experts. Activation ordering defaults to weight ordering, which keeps group boundaries in the original column space and stores no `weight_g_idx`. Group INT4 and group INT8 weights cannot be mixed in one checkpoint, and group weights cannot be mixed with per-channel or block weights. `INT4_BLOCK` / `INT8_BLOCK` remain supported when an explicit recipe declares them, but are not exposed as output formats.

The INT4 checkpoint uses QLean's A8W4 weight layout, packed by the existing pure PyTorch backend; no ACExt dependency or native extension is required for packing. The local packer requires nonempty 2D INT4 expert weights with N and K multiples of 64; companion channel INT8 weights do not have this restriction. It is not a standard compressed-tensors W4A16 checkpoint and is not reloadable with standard compressed-tensors/Hugging Face INT4 loaders. See `examples/gptq_quantization.py` for a BF16 MoE Python example with calibration data. GPTQ auto-routing uses the packaged `gptq/_arch/default.yaml` when no model-specific or architecture recipe matches; GPTQ needs calibration data, unlike the data-free engine conversions. All `GPTQ_*` formats require the optional extra (`pip install ".[gptq]"`, which pulls compressed-tensors>=0.18 / torch>=2.10 and transformers>=5.15); without it the core install still imports and data-free conversions still run, but a GPTQ route raises `ImportError` naming the extra.

`GPTQ_INT8` automatically selects symmetric, static, per-channel integer INT8 Linear weights, with no `group_size` or activation quantization. Unselected weights remain BF16 when loaded in BF16. The final custom checkpoint is marked `w8a8_int8`: signed INT8 `.weight` tensors of shape `[N, K]` and FP32 `.weight_scale` tensors of shape `[N, 1]`, with no INT4-style multiple-of-64 restriction. It requires a compatible custom W8A8 INT8 consumer that dynamically quantizes activations at runtime, not standard compressed-tensors/Hugging Face weight reload. `GPTQ_MXFP4` is the only GPTQ MXFP4 output name and keeps the existing grouped MXFP4 contract: the checkpoint is marked `mxfp4` with no `config_groups`/`format`, `.weight` tensors are packed uint8 of shape `[N, K/2]` and `.weight_scale` tensors are E8M0 uint8 of shape `[N, K/32]` (one power-of-2 scale per 32-element group). Like `w8a8_int8`, it requires a compatible custom MXFP4 consumer, not standard compressed-tensors/Hugging Face weight reload. The packaged default still uses `offload_hessians: false`.

## Output Directory Rules

`--save_path` must not be the same as, or nested inside/around, `--model_path`, including through symlinks; GPTQ checks this before calibration. GeneralQuantEngine conversions resume by default; any config change alters the fingerprint and reprocesses the affected shards.

Custom A8W4 (`GPTQ_INT4`) and channel INT8 (`GPTQ_INT8`, final format `w8a8_int8`) export require a fresh weights directory. An absent directory or an existing directory without checkpoint weight files or weight-index artifacts is allowed; an existing checkpoint cannot be resumed or overwritten. This also applies when an explicit recipe selects either custom contract, but does not change the MXFP4 or group/block save paths. Use a new output directory rather than deleting or overwriting source weights.
