# QLean

QLean is a model weight quantization, dequantization, and mixed-precision conversion tool. Specify a target format with `--output-format` and convert in one command.

**Version:** v0.3.0

Pre-quantized model weights are available on [ModelScope (T-HEAD)](https://modelscope.cn/organization/T-HEAD).

## Capability Matrix

| Source Formats | Target Format | Description |
|:---|:---|:---|
| BF16 / FP8_BLOCK / MXFP4 / INT4 | INT8_W8A8 | Weight: INT8, static, Per-Channel<br>Activation: INT8, dynamic, Per-Token |
| BF16 / MXFP4 / INT4 | FP8_BLOCK | Weight: FP8, static, Per-Block (128×128)<br>Activation: FP8, dynamic, Per-Group (128) |
| BF16 / FP8_BLOCK / MXFP4 / INT4 | FP8_CHANNEL | Weight: FP8, static, Per-Channel<br>Activation: FP8, dynamic, Per-Token |
| BF16 / FP8_BLOCK / INT4 / NVFP4 | MXFP4 | Weight: OCP MXFP4, static<br>Activation: OCP MXFP4, dynamic |
| FP8_BLOCK / MXFP4 / INT4 / NVFP4 | BF16 | Dequantized to BF16 |

## Installation

### Requirements

- PPU hardware with a compatible driver and SDK/runtime (a Docker image matching the PPU SDK version is recommended).
- Python >=3.10,<3.15.
- Compatible PPU builds of PyTorch (`torch>=2.7,<2.14`) and Triton (`triton>=3.4,<3.8`).

### Install

Install QLean in the prepared PPU environment:

```bash
git clone https://github.com/t-head/QLean-for-SAIL.git && cd QLean-for-SAIL
pip install .
```

The core install covers the data-free conversions and keeps the wide
`torch` / `transformers` ranges above.

Calibration-based GPTQ (`--output-format GPTQ_*`) is an optional extra with
narrower floors, because it needs the compressed-tensors offload/distributed
subpackages and the transformers v5 weight-conversion APIs:

```bash
pip install ".[gptq]"   # compressed-tensors>=0.18 (torch>=2.10), transformers>=5.15
```

Without the extra, `qlean` still imports and the data-free conversions still run;
reaching a GPTQ route raises an `ImportError` that names the extra. The PPU
`torch 2.9` build predates `torch>=2.10`, so GPTQ currently requires a non-PPU
environment or a newer PPU torch build.

## Quick Start

```bash
qlean --model_name Qwen/Qwen3-235B-A22B \
      --model_path /path/to/Qwen3-235B-A22B \
      --save_path /path/to/Qwen3-235B-A22B-INT8
```

- `--model_name` must match the official HuggingFace model stub.
- `--model_path` and `--save_path` are local directories.
- Omitting `--output-format` defaults to `INT8_W8A8`.
- The output holds the converted model plus `qlean_config.json`, a record of the conversion.

Other common invocations:

```bash
# FP8 source -> INT8, source format detected automatically
qlean --model_name deepseek-ai/DeepSeek-V3.2 \
      --model_path /path/to/DeepSeek-V3.2 \
      --save_path /path/to/DeepSeek-V3.2-INT8

# change the target format
qlean --model_name Qwen/Qwen3.5-397B-A17B \
      --model_path /path/to/Qwen3.5-397B-A17B \
      --save_path /path/to/Qwen3.5-397B-A17B-MXFP4 \
      --output-format MXFP4

# custom recipe; overrides --output-format and built-in recipes
qlean --model_name Qwen/Qwen3-8B \
      --model_path /path/to/Qwen3-8B \
      --save_path /path/to/Qwen3-8B-custom \
      --recipe /path/to/recipe.yaml
```

Every flag, format name, and alias: [docs/cli.md](docs/cli.md).

## Mode: Auto vs Manual

A recipe is a YAML file that tells QLean exactly what to do: which engine (modifier)
to run, which tensors to convert, and each one's source and target format.
Auto mode picks a built-in recipe for you; Manual mode lets you use your own.

| Mode | Trigger | Behavior | When to Use |
|:---|:---|:---|:---|
| **Auto** (default) | `--output-format` | Detects the source format, matches a built-in recipe | Model has a built-in recipe |
| **Manual** | `--recipe path/to/recipe.yaml` | Follows the custom recipe; `--output-format` and built-in recipes are ignored | Mixed sources, custom layouts, tailored strategies |

- `qlean --list-models` lists the models with built-in recipes. For those, omit `--recipe`. 
- Unlisted models fall back to an architecture-level preset or a default recipe; write your own for best results. You can refer to [docs/manual.md](docs/manual.md).

## Verification

`--verify bound` (or `verify: bound` in a recipe) compares supported quantized tensors' reconstruction errors against the theoretical bounds of their target formats and writes reports to `<save_path>/qlean_verify/`. Numerical violations do not stop conversion. Verification is not a substitute for end-to-end evaluation.

```
[verify] Coverage: 141/141 shards
[verify] Result: PASSED | Checked 141/141 shards, 59044 rows = ok 59044 / violation 0 / skipped 0 / error 0
```

It applies only to `GeneralQuantEngineModifier` recipes. It covers 2D `W8A8`, `FP8_DYNAMIC` and `MXFP4` operations. Unsupported schemes are recorded as `skipped`. For the report layout, status semantics, and interaction with resume, see [docs/verification.md](docs/verification.md).

## Documentation

| | |
|:---|:---|
| [docs/cli.md](docs/cli.md) | Flags, helper commands, format names and aliases |
| [docs/manual.md](docs/manual.md) | Manual mode: recipe structure, modifier fields, region rules, examples |
| [docs/verification.md](docs/verification.md) | Reports, statuses, resume behaviour, troubleshooting |
| [docs/faq.md](docs/faq.md) | Common failures and how to read the logs |

## Contact

- Questions, bug reports, and feature requests: [GitHub Issues](https://github.com/t-head/QLean-for-SAIL/issues)

## License

Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.

QLean is a model weight quantization, dequantization and conversion tool developed by T-Head and licensed under the Apache License (Version 2.0). See [LICENSE](LICENSE).

This product contains third-party components under other open source licenses. See [NOTICE](NOTICE) for more information.
