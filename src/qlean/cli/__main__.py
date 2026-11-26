# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import re
import sys

def setup_logging(level):
    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(level=level, format=fmt)

_INPUT_FORMAT_DEFS = {
    "FP8":   ["FP8"],
    "NVFP4": ["NVFP4"],
    "BF16":  ["BF16"],
}

INPUT_FORMAT_ALIASES = {}
for canonical, aliases in _INPUT_FORMAT_DEFS.items():
    for alias in aliases:
        for variant in (alias, alias.upper(), alias.lower()):
            INPUT_FORMAT_ALIASES[variant] = canonical

def parse_input_format(value: str) -> str:
    canonical = INPUT_FORMAT_ALIASES.get(value)
    if canonical is None:
        raise argparse.ArgumentTypeError(
            f"Unknown input format '{value}'. Supported: FP8, NVF4, BF16"
        )
    return canonical

_OUTPUT_FORMAT_DEFS = {
    "INT8_W8A8":   ["W8A8", "INT8", "INT8_W8A8"],
    "FP8_BLOCK":   ["FP8-BLOCK", "FP8_BLOCK"],
    "FP8_DYNAMIC": ["FP8-DYNAMIC", "FP8_DYNAMIC", "FP8-CHANNEL", "FP8_CHANNEL"],
    "MXFP4":       ["MXFP4"],
    "GPTQ_INT4":   ["GPTQ_INT4"],
    "GPTQ_INT8":   ["GPTQ_INT8"],
    "GPTQ_MXFP4":  ["GPTQ_MXFP4"],
    "GPTQ_INT4_GROUP": ["GPTQ_INT4_GROUP"],
    "GPTQ_INT8_GROUP": ["GPTQ_INT8_GROUP"],
    "BF16":        ["BF16"],
}

OUTPUT_FORMAT_ALIASES = {}
for canonical, aliases in _OUTPUT_FORMAT_DEFS.items():
    for alias in aliases:
        for variant in (alias, alias.upper(), alias.lower()):
            OUTPUT_FORMAT_ALIASES[variant] = canonical

def parse_output_format(value: str) -> str:
    canonical = OUTPUT_FORMAT_ALIASES.get(value)
    if canonical is None:
        # Keep help and argument parsing independent of torch/modifier imports.
        from qlean.core.routes import validate_gptq_output_format

        try:
            validate_gptq_output_format(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        supported = sorted(set(OUTPUT_FORMAT_ALIASES.values()))
        aliases = {v: [k for k, v2 in OUTPUT_FORMAT_ALIASES.items() if v2 == v] for v in supported}
        lines = [f"  {fmt}: {', '.join(a)}" for fmt, a in aliases.items()]
        raise argparse.ArgumentTypeError(
            f"Unknown output format '{value}'\nSupported formats:\n" + "\n".join(lines)
        )
    return canonical

_DESCRIPTION = """\
\nQLean - A toolkit for quantizing and converting large language models.

How it works:
  QLean auto-detects your model's current format (BF16, FP8, NVFP4),
  then applies the necessary conversion to reach your target format.
  If the source model is already quantized (e.g. FP8), QLean will
  dequantize first, then re-quantize to the target format."""

_USAGE = """\
qlean --model_name MODEL_NAME --model_path MODEL_PATH
             --save_path SAVE_PATH [options]"""

def _print_examples():
    """Print usage examples."""
    print("""\
\nExamples:

  # 1. Quantize a BF16 model to INT8 (default output format)
  qlean --model_name Qwen/Qwen3-235B-A22B \\
        --model_path /path/to/Qwen3-235B-A22B \\
        --save_path /path/to/Qwen3-235B-A22B-INT8

  # 2. FP8 model → INT8 (input format auto-detected from model files)
  qlean --model_name deepseek-ai/DeepSeek-V3.2 \\
        --model_path /path/to/DeepSeek-V3.2 \\
        --save_path /path/to/DeepSeek-V3.2-INT8

  # 3. Choose a different output format with --output-format
  qlean --model_name Qwen/Qwen3-30B-A3B \\
        --model_path /path/to/Qwen3-30B-A3B \\
        --save_path /path/to/Qwen3-30B-A3B-FP8Block \\
        --output-format FP8_BLOCK

  # 4. Use a custom recipe file (overrides built-in recipe)
  qlean --model_name Qwen/Qwen3-8B \\
        --model_path /path/to/Qwen3-8B \\
        --save_path /path/to/Qwen3-8B-custom \\
        --recipe /path/to/recipe.yaml
""")

def _print_formats():
    """Print engine conversions plus the calibration-based GPTQ routes."""
    from qlean.core.auto_recipe import AUTO_CONVERSIONS
    from qlean.core.routes import GPTQ_OUTPUT_PRESETS, MODIFIER_REPLACEMENT
    from collections import defaultdict

    # Keep every engine conversion; add canonical GPTQ output names only.
    conversions = defaultdict(set)
    for (inp, out) in AUTO_CONVERSIONS:
        conversions[inp].add(out)
    for inp, out in MODIFIER_REPLACEMENT:
        if out in GPTQ_OUTPUT_PRESETS:
            conversions[inp].add(OUTPUT_FORMAT_ALIASES[out])

    print("\nSupported conversions:\n")
    print(f"  {'Input':<10} {'Output Formats'}")
    print(f"  {'─' * 10} {'─' * 40}")
    for inp in sorted(conversions):
        outputs = ", ".join(sorted(conversions[inp]))
        print(f"  {inp:<10} {outputs}")

    output_aliases = {c: [a for a in al if a != c] for c, al in _OUTPUT_FORMAT_DEFS.items()}
    if any(output_aliases.values()):
        print("\nOutput format aliases:")
        for canonical, other in output_aliases.items():
            if other:
                print(f"  {canonical}: {', '.join(other)}")

    input_aliases = {c: [a for a in al if a != c] for c, al in _INPUT_FORMAT_DEFS.items()}
    if any(input_aliases.values()):
        print("\nInput format aliases:")
        for canonical, other in input_aliases.items():
            if other:
                print(f"  {canonical}: {', '.join(other)}")
    print()

def _natural_sort_key(s):
    """Return a key for natural sorting (e.g. 8B < 30B < 235B)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def _print_models():
    """Print models with optimized recipes, grouped by output format."""
    from importlib.resources import files
    from qlean.core.routes import RECIPE_DIR_MAP

    # Build reverse map: dir_name -> list of output formats, excluding BF16
    dir_to_fmts = {}
    for fmt, dir_name in sorted(RECIPE_DIR_MAP.items()):
        if fmt == "BF16":
            continue
        dir_to_fmts.setdefault(dir_name, []).append(fmt)

    print("\nModels with dedicated recipes")
    print("(grouped by supported output format):\n")
    for dir_name, fmts in sorted(dir_to_fmts.items()):
        label = " / ".join(sorted(fmts))
        print(f"  {label}:")

        base = files("qlean.configs").joinpath(dir_name)
        try:
            model_names = []
            for item in base.iterdir():
                name = item.name
                if name.endswith(".yaml") and not name.startswith("_"):
                    model_names.append(name.removesuffix('.yaml'))
        except (FileNotFoundError, TypeError):
            model_names = []

        # Group by brand (first segment before '-'), sort naturally within
        brand_map = {}
        for name in model_names:
            brand = name.split("-")[0]
            brand_map.setdefault(brand, []).append(name)

        max_brand_len = max((len(b) for b in brand_map), default=0)
        for brand in sorted(brand_map):
            models = sorted(brand_map[brand], key=_natural_sort_key)
            padding = ' ' * (max_brand_len - len(brand))
            print(f"    {brand}:{padding}  {', '.join(models)}")
        print()

    print("Models not listed may still work with architecture-level or default")
    print("recipes, which may not be fully optimized for those models.")
    print()
    print("Use --list-models MODEL_NAME to check support for any model.\n")

def _print_model_recipe(model_name: str):
    """Check whether a model has dedicated recipe support."""
    from importlib.resources import files
    from qlean.core.routes import RECIPE_DIR_MAP

    short_name = model_name.split("/")[-1]

    # Build reverse map: dir_name -> list of output formats, excluding BF16
    dir_to_fmts = {}
    for fmt, dir_name in sorted(RECIPE_DIR_MAP.items()):
        if fmt == "BF16":
            continue
        dir_to_fmts.setdefault(dir_name, []).append(fmt)

    matched_fmts = []
    all_fmts = []
    for dir_name, fmts in sorted(dir_to_fmts.items()):
        all_fmts.extend(fmts)
        base = files("qlean.configs").joinpath(dir_name)
        exact = base / f"{short_name}.yaml"
        if exact.is_file():
            matched_fmts.extend(fmts)

    if matched_fmts:
        label = " / ".join(sorted(matched_fmts))
        print(f"\n'{short_name}' has a dedicated recipe for: {label}")
        missing_fmts = sorted(set(all_fmts) - set(matched_fmts))
        if missing_fmts:
            missing_label = " / ".join(missing_fmts)
            print(f"\nNo dedicated recipe for: {missing_label}")
            print("These formats will use a fallback recipe,")
            print("which may not be fully optimized for this model.")
        print()
    else:
        print(f"\nNo dedicated recipe for '{short_name}'.")
        print("QLean will use an architecture-level or default recipe at runtime,")
        print("which may not be fully optimized for this model.\n")

# Options whose argparse default is None so a model profile can supply the value
# the reference torchrun script uses. An explicit user value always wins.
_MODEL_OPTION_DEFAULTS = {
    "cpu_offload_gb": 300.0,
    "gptq_export_workers": 2,
    "num_calibration_rounds": 1,
    "calibration_padding": "none",
}


def _apply_model_defaults(args):
    """Fill options the user did not pass, before the dataclasses validate them.

    Must run before ``parse_to_dataclass``: these flags default to None so a model
    profile can supply the reference script's value, and the dataclass validators
    reject None (e.g. num_calibration_rounds must be a positive integer).
    """
    from qlean.models.pretrained.model_registry import resolve_model_defaults

    log = logging.getLogger(__name__)
    registered = resolve_model_defaults(
        getattr(args, "model_name", None), getattr(args, "model_path", None)
    )
    for name, fallback in _MODEL_OPTION_DEFAULTS.items():
        if getattr(args, name, None) is not None:
            continue
        value = registered.get(name, fallback)
        setattr(args, name, value)
        if name in registered:
            log.info("Using %s=%r registered for %s", name, value, args.model_name)


def main():
    parser = argparse.ArgumentParser(
        prog="qlean",
        usage=_USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_DESCRIPTION,
        add_help=False,
    )

    required_group = parser.add_argument_group("Required Arguments")
    required_group.add_argument(
        "--model_name",
        type=str,
        required=False,
        default=None,
        metavar="MODEL_NAME",
        help="Model identifier (e.g. 'Qwen/Qwen3-235B-A22B')",
    )
    required_group.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=None,
        metavar="MODEL_PATH",
        help="Path to the source model directory",
    )
    required_group.add_argument(
        "--save_path",
        type=str,
        required=False,
        default=None,
        metavar="SAVE_PATH",
        help="Path to save the output model",
    )

    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "--output-format",
        type=parse_output_format,
        default="INT8_W8A8",
        metavar="FORMAT",
        help="Target output format (default: INT8_W8A8)",
    )
    options_group.add_argument(
        "--input-format",
        type=parse_input_format,
        default=None,
        metavar="FORMAT",
        help="Source model format (auto-detected if omitted)",
    )
    options_group.add_argument(
        "--recipe",
        type=str,
        required=False,
        help="Path to a custom recipe YAML file",
    )
    options_group.add_argument(
        "--mix_path",
        type=str,
        required=False,
        metavar="MIX_PATH",
        help="Path to save the mixed-precision model",
    )
    options_group.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Logging verbosity (default: INFO)",
    )
    options_group.add_argument(
        "--verify",
        choices=["off", "bound"],
        default=None,
        help="Override GeneralQuantEngineModifier.verify (off|bound); "
             "omitted = follow recipe",
    )

    calibration_group = parser.add_argument_group("Calibration Data (GPTQ)")
    calibration_group.add_argument(
        "--dataset_name", type=str, default=None, metavar="ID",
        help="Hub dataset id for calibration (e.g. HuggingFaceH4/ultrachat_200k)",
    )
    calibration_group.add_argument(
        "--dataset_path", type=str, default=None, metavar="PATH",
        help="Local dataset path for calibration",
    )
    calibration_group.add_argument(
        "--dataset_config", type=str, default=None, metavar="NAME",
        help="Dataset config/subset (e.g. wikitext-2-raw-v1)",
    )
    calibration_group.add_argument(
        "--dataset_split", type=str, default="train_sft", metavar="SPLIT",
        help="Split to partition per rank (default: train_sft)",
    )
    calibration_group.add_argument(
        "--num_calibration_samples", type=int, default=256, metavar="N",
        help="Total calibration samples across all ranks (default: 256)",
    )
    calibration_group.add_argument(
        "--max_sequence_length", type=int, default=2048, metavar="L",
        help="Tokenization truncation length (default: 2048)",
    )
    calibration_group.add_argument(
        "--calibration_padding", type=str, default=None, choices=["none", "max_length"],
        metavar="MODE",
        help="Tokenization padding: 'none' keeps ragged lengths, 'max_length' pads to "
             "--max_sequence_length (default: none, or the model's registered default)",
    )
    calibration_group.add_argument(
        "--batch_size", type=int, default=1, metavar="N",
        help="Calibration batch size (default: 1)",
    )
    calibration_group.add_argument(
        "--seed", type=int, default=42, help="Calibration shuffle seed (default: 42)",
    )
    calibration_group.add_argument(
        "--messages_column", type=str, default="messages", metavar="COL",
        help="Chat-messages column rendered via apply_chat_template (default: messages)",
    )
    calibration_group.add_argument(
        "--text_column", type=str, default="text", metavar="COL",
        help="Plain-text column used when no messages column exists (default: text)",
    )

    gptq_group = parser.add_argument_group("GPTQ Model Loading")
    gptq_group.add_argument(
        "--offload_folder", type=str, default=None, metavar="PATH",
        help="Writable disk-offload directory; required for GPTQ and must remain "
             "available through calibration and export",
    )
    gptq_group.add_argument(
        "--device_map", type=str, default="auto_offload", metavar="MAP",
        help="from_pretrained device_map (default: auto_offload)",
    )
    gptq_group.add_argument(
        "--cpu_offload_gb", type=float, default=None, metavar="GB",
        help="CPU offload budget in GB (default: 300, or the model's registered default)",
    )
    gptq_group.add_argument(
        "--trust_remote_code", action=argparse.BooleanOptionalAction, default=True,
        help="Allow custom checkpoint code (default: enabled; --no-trust_remote_code to disable)",
    )
    gptq_group.add_argument(
        "--max_shard_size", type=str, default="10GB", metavar="SIZE",
        help="Output shard size (default: 10GB)",
    )
    gptq_group.add_argument(
        "--gptq_export_workers", type=int, default=None, metavar="N",
        help="Threads for INT8/MXFP4 export (default: 2, or the model's registered default)",
    )
    gptq_group.add_argument(
        "--gptq_int4_export_device", type=str, default="auto", metavar="DEV",
        help="A8W4 packing device: auto|cpu|cuda[:N] (default: auto)",
    )

    dist_group = parser.add_argument_group("Distributed GPTQ")
    dist_group.add_argument(
        "--nproc_per_node", type=int, default=1, metavar="N",
        help="Processes per node; >1 re-execs qlean under torchrun (default: 1)",
    )
    dist_group.add_argument(
        "--nnodes", type=int, default=1, metavar="N", help="Node count (default: 1)",
    )
    dist_group.add_argument(
        "--rdzv_endpoint", type=str, default=None, metavar="HOST:PORT",
        help="Rendezvous endpoint for multi-node runs",
    )
    dist_group.add_argument(
        "--max_restarts", type=int, default=0, metavar="N",
        help="torchrun worker restarts; keep 0 for stateful quantization (default: 0)",
    )
    dist_group.add_argument(
        "--num_calibration_rounds", type=int, default=None, metavar="N",
        help="Split each layer's targets into N rounds to bound Hessian memory "
             "(default: 1, or the model's registered default)",
    )
    dist_group.add_argument(
        "--propagate_error", action=argparse.BooleanOptionalAction, default=True,
        help="Re-forward quantized layers to propagate error (default: enabled)",
    )

    options_group.add_argument(
        "--other_args",
        type=str,
        required=False,
        help=argparse.SUPPRESS,
    )

    info_group = parser.add_argument_group("Info Commands")
    info_group.add_argument(
        "-h", "--help",
        action="help",
        help="Show this help message and exit",
    )
    info_group.add_argument(
        "--list-formats",
        action="store_true",
        help="Show all supported format conversions and aliases",
    )
    info_group.add_argument(
        "--list-models",
        nargs="?",
        const=True,
        default=None,
        metavar="MODEL",
        help="List supported models, or show a model's recipe",
    )
    info_group.add_argument(
        "--examples",
        action="store_true",
        help="Show usage examples",
    )

    args = parser.parse_args()

    # Handle info commands (no model args required)
    if getattr(args, 'list_formats', False):
        _print_formats()
        return
    if getattr(args, 'list_models', None) is not None:
        if args.list_models is True:
            _print_models()
        else:
            _print_model_recipe(args.list_models)
        return
    if getattr(args, 'examples', False):
        _print_examples()
        return

    # Validate required args for quantization
    missing = []
    for name in ('model_name', 'model_path', 'save_path'):
        if getattr(args, name) is None:
            missing.append(f'--{name}')
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Application start")

    # Load heavy deps (torch etc.) only when entering the quantization flow;
    # info-only commands stay lightweight.
    from qlean.utils import parse_to_dataclass
    from qlean.entrypoints import post_train

    _apply_model_defaults(args)
    model_args, dataset_args, other_args = parse_to_dataclass(args)

    from qlean.cli.launcher import launch_torchrun, needs_launch

    if needs_launch(other_args):
        raise SystemExit(launch_torchrun(other_args, sys.argv[1:]))

    post_train(model_args, dataset_args, other_args)

if __name__ == "__main__":
    main()