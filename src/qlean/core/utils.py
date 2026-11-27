# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys, os
import copy
import yaml, json
import subprocess
from glob import glob
from typing import List, Tuple
from safetensors import safe_open
from importlib.resources import files
from pathlib import Path

from qlean.args import ModelArguments
from qlean.configs import load_env_rules
from qlean.core.routes import (
    GPTQ_EXPERT_TARGET,
    GPTQ_INT4_PRESETS,
    GPTQ_OUTPUT_PRESETS,
    GPTQ_WEIGHT_CONFIGS,
    MODIFIER_REPLACEMENT,
    RECIPE_DIR_MAP,
    SCHEME_MAP,
    validate_gptq_output_format,
)
from qlean.modifiers import ModifierFactory, Modifier

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "install_dependencies",
    "preprocess_model",
    "get_modifiers_from_recipe",
    "remove_unsupported_fields",
    "detect_input_format",
    "resolve_recipe",
    "save_qlean_metadata",
]

def install_dependencies(model_name: str) -> str:
    config = load_env_rules()
    name, packages = get_dependencies(config, model_name)
    install_packages(packages)
    return name

def get_dependencies(config: dict, model_name: str) -> Tuple[str, List[str]]:
    # Match the configured model names; use fallback packages when no rule matches.
    rules = config.get("rules", [])
    for rule in rules:
        if model_name in rule["name"]:
            logger.info(f"Support load model {rule['name']}\n")
            name = rule["name"].rsplit("/", 1)[-1]
            return (name, rule.get("pip_packages", []))

    logger.info(f"Not support load model {model_name}, process as day0")
    fallback = config.get("fallback", [])
    return (fallback["name"], fallback.get("pip_packages", []))

def install_packages(packages: List[str]):
    for package in packages:
        logger.info(f"Installing package {package}")
        cmd = [sys.executable, "-m", "pip", "install", "--progress-bar", "on", package]
        result = subprocess.run(cmd)  # , stdout=subprocess.DEVNULL)

        if result.returncode == 0:
            logger.info(f"Install success: {package}")
        else:
            logger.error(f"Install fail: {package}, {result.stderr}")

def preprocess_model(model):
    return model

def update_scheme(data, new_value):
    if isinstance(data, dict):
        for key, value in data.items():
            # MXFP4 recipes define fixed per-region mixed-precision targets.
            update_dst = (
                key == "dst" and new_value != "MXFP4"
                and isinstance(value, str)
                and value.lower() not in ("bf16", "passthrough")
            )
            if key in ("scheme", "primary_scheme") or update_dst:
                data[key] = new_value
                logger.info(f"Updated '{key}' from '{value}' to '{new_value}'")
            else:
                update_scheme(value, new_value)
    elif isinstance(data, list):
        for item in data:
            update_scheme(item, new_value)


def inject_verify(data, value) -> bool:
    """Write the CLI --verify value into every GeneralQuantEngineModifier block of a
    recipe dict. Only that modifier is touched: others have no `verify` field, so
    injecting one would fail Pydantic validation. Returns True if any block was hit.
    """
    hit = False
    if isinstance(data, dict):
        for key, sub in data.items():
            if key == "GeneralQuantEngineModifier":
                if not isinstance(sub, dict):
                    data[key] = sub = {}
                sub["verify"] = value
                hit = True
            elif inject_verify(sub, value):
                hit = True
    elif isinstance(data, list):
        for item in data:
            hit = inject_verify(item, value) or hit
    return hit


def _try_inject_verify(recipe_dict, model_args) -> None:
    """Inject --verify when given; warn if the recipe has no GeneralQuantEngineModifier.

    Shared by the three branches of get_modifiers_from_recipe.
    """
    if model_args.verify is not None:
        if not inject_verify(recipe_dict, model_args.verify):
            logger.warning(
                f"--verify {model_args.verify} was given, but the recipe has no "
                f"GeneralQuantEngineModifier; ignoring"
            )


def _recipe_has_modifier(recipe_dict: dict, mod_type: str) -> bool:
    for stage_key, stage_val in recipe_dict.items():
        if not (stage_key.endswith("_stage") and isinstance(stage_val, dict)):
            continue
        for group_key, group_val in stage_val.items():
            if group_key.endswith("_modifiers") and isinstance(group_val, dict):
                if mod_type in group_val:
                    return True
    return False


def _retarget_recipe_to_bf16(recipe_dict: dict, input_format: str, recipe) -> None:
    """Make a resolved recipe actually emit BF16, or raise if it cannot."""
    from qlean.core.auto_recipe import retarget_engine_recipe_to_bf16

    if input_format == "BF16":
        raise ValueError(
            "BF16 → BF16 is not a conversion: the weights are already BF16. "
            "Drop --output-format BF16, or pass --input-format with the real "
            "source format if detection picked the wrong one."
        )

    rewritten = retarget_engine_recipe_to_bf16(recipe_dict)
    if rewritten:
        logger.info(
            f"Retargeted {rewritten} GeneralQuantEngineModifier(s) in {recipe} "
            f"to dequantize {input_format} to bf16"
        )
        return

    if ((input_format, "BF16") in MODIFIER_REPLACEMENT
            and _recipe_has_modifier(recipe_dict, "generalDay0Modifier")):
        # The route table replaces generalDay0Modifier with the dequant chain.
        return

    raise ValueError(
        f"--output-format BF16 cannot be honored by recipe {recipe}: it has no "
        f"GeneralQuantEngineModifier to retarget, and no generalDay0Modifier for "
        f"the {input_format} → BF16 route to replace. Running it unchanged would "
        f"produce the recipe's own target format instead of BF16. Pass an explicit "
        f"--recipe that dequantizes this model."
    )


def detect_input_format(model_path: str) -> str:
    """
    Detect the weight format of a model, e.g. "FP8" / "NVFP4" / "BF16".
    Three levels: config.json/hf_quant_config.json → index.json → safetensor keys.
    """
    # Level 1: quantization_config
    quant_method = _read_quant_method(model_path)
    if quant_method in ("fp8",):
        return "FP8"
    if quant_method in ("nvfp4",):
        return "NVFP4"
    if quant_method in ("compressed-tensors", "gptq", "awq", "bitsandbytes"):
        return quant_method.upper()  # internal format, absent from the route table; auto mode rejects it

    # Level 2: weight_map keys in index.json
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        weight_map = json.load(open(index_file)).get("weight_map", {})
        fmt = _detect_from_keys(weight_map.keys())
        if fmt:
            return fmt

    # Level 3: scan the keys of every safetensors file
    st_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    for st_file in st_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            fmt = _detect_from_keys(f.keys())
            if fmt:
                return fmt

    return "BF16"

def _read_quant_method(model_path: str) -> str:
    for filename in ("config.json", "hf_quant_config.json"):
        filepath = os.path.join(model_path, filename)
        if not os.path.exists(filepath):
            continue
        data = json.load(open(filepath))
        result = _find_quant_method(data)
        if result:
            return result.lower()
    return ""

def _find_quant_method(obj: dict, depth: int = 0) -> str:
    """
    Find the quantization method inside a config object.
    Supports HuggingFace, NVIDIA ModelOpt, and plain quantization config dicts.
    """
    if not isinstance(obj, dict):
        return ""

    # 1. Common fields at this level (hf_quant_config.json is loaded flat).
    # Check the specific `algo` key before the generic `method` one.
    for key in ["quant_algo", "quant_method", "quantization_type"]:
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val

    # 2. HuggingFace standard nesting: quantization_config
    qconfig = obj.get("quantization_config")
    if isinstance(qconfig, dict):
        for key in ["quant_algo", "quant_method"]:
            val = qconfig.get(key)
            if isinstance(val, str) and val:
                return val

    # 3. NVIDIA/ModelOpt style: quantization (dict or string)
    quantization = obj.get("quantization")
    if isinstance(quantization, dict):
        for key in ["quant_algo", "quant_method"]:
            val = quantization.get(key)
            if isinstance(val, str) and val:
                return val
    elif isinstance(quantization, str) and quantization:
        # vLLM style: "quantization": "fp8"
        return quantization

    # 4. GGUF-style metadata, when present
    gguf_type = obj.get("gguf", {}).get("quantization_type") if isinstance(obj.get("gguf"), dict) else None
    if isinstance(gguf_type, str) and gguf_type:
        return gguf_type

    # 5. Recurse into nested dicts as a last resort.
    # Depth is capped to avoid matching unrelated deep config.
    if depth < 3:
        for key, value in obj.items():
            if isinstance(value, dict):
                result = _find_quant_method(value, depth + 1)
                if result:
                    return result

    return ""

def _detect_from_keys(keys) -> str | None:
    has_scale_inv = False
    has_weight_scale = False
    has_weight_global_scale = False
    has_input_scale = False

    for key in keys:
        if key.endswith("scale_inv"):
            has_scale_inv = True
        elif key.endswith(".weight_scale"):
            has_weight_scale = True
        elif key.endswith((".weight_scale_2", ".weight_global_scale")):
            has_weight_global_scale = True
        elif key.endswith(".input_scale"):
            has_input_scale = True

    # NVFP4: local scale, global scale, and input scale must all be present.
    if has_weight_scale and has_weight_global_scale and has_input_scale:
        return "NVFP4"
    # FP8: any key carrying the .scale_inv suffix
    if has_scale_inv:
        return "FP8"
    return None

def apply_gptq_output_format(recipe_dict: dict, output_format: str) -> str | None:
    """Rewrite an automatically resolved recipe; explicit user recipes bypass this."""
    validate_gptq_output_format(output_format)
    preset = GPTQ_OUTPUT_PRESETS.get(output_format)
    if preset is None:
        return None

    replacement_weights = GPTQ_WEIGHT_CONFIGS[preset]
    updated = 0

    def visit(value):
        nonlocal updated
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "GPTQModifier" and (child is None or isinstance(child, dict)):
                    if child is None:
                        # An empty `GPTQModifier:` body is a legal recipe spelling
                        # (get_modifiers_from_dict treats None as {}). Materialize
                        # it, otherwise the preset written below would be lost.
                        child = {}
                        value[key] = child
                    config_groups = child.get("config_groups")
                    if isinstance(config_groups, dict) and config_groups:
                        child.pop("scheme", None)
                        for config_group in config_groups.values():
                            if isinstance(config_group, dict):
                                weights = config_group.get("weights") or {}
                                replacement = copy.deepcopy(replacement_weights)
                                if "actorder" in weights:
                                    replacement["actorder"] = weights["actorder"]
                                config_group["weights"] = replacement
                                config_group.pop("input_activations", None)
                                config_group.pop("output_activations", None)
                                updated += 1
                    else:
                        child.pop("config_groups", None)
                        child["scheme"] = preset
                        if preset in GPTQ_INT4_PRESETS:
                            child["targets"] = [GPTQ_EXPERT_TARGET]
                        else:
                            child.setdefault("targets", ["Linear"])
                        updated += 1
                else:
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(recipe_dict)
    if updated == 0:
        raise ValueError(
            f"GPTQ output format '{output_format}' requires a GPTQModifier recipe"
        )

    logger.info("Applied GPTQ preset %s to %d config group(s)", preset, updated)
    return preset

def resolve_recipe(output_format: str, model_name: str, model_path: str) -> Path | None:
    """Match model/architecture recipes, then the GPTQ-only packaged default.

    None allows non-GPTQ callers to generate a GeneralQuantEngine recipe.
    """
    validate_gptq_output_format(output_format)
    recipe_dir = RECIPE_DIR_MAP.get(output_format)
    if recipe_dir is None:
        raise ValueError(f"No recipe directory configured for output format '{output_format}'")

    base = files("qlean.configs").joinpath(recipe_dir)
    short_name = model_name.split("/")[-1]

    # Level 1: exact model name
    exact = base / f"{short_name}.yaml"
    if exact.is_file():
        logger.info(f"Matched exact recipe: {exact}")
        return exact

    # Level 2: architecture fallback
    arch = _get_architecture(model_path)
    if arch:
        arch_recipe = base / "_arch" / f"{arch}.yaml"
        if arch_recipe.is_file():
            logger.info(f"Matched architecture recipe: {arch_recipe} (arch={arch})")
            return arch_recipe

    # GPTQ is not supported by the engine-generated fallback.
    if output_format in GPTQ_OUTPUT_PRESETS:
        default = base / "_arch" / "default.yaml"
        if default.is_file():
            logger.info(f"Using default GPTQ recipe: {default}")
            return default
        raise ValueError(f"No GPTQ default recipe found: {default}")

    # Non-GPTQ formats keep the GeneralQuantEngine-generated fallback.
    return None

def _get_architecture(model_path: str) -> str | None:
    """Infer the architecture name from the `architectures` field of config.json."""
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        return None

    with open(config_file) as f:
        config = json.load(f)

    # Prefer architectures, e.g. ["DeepseekV3ForCausalLM"]
    archs = config.get("architectures", [])
    if archs:
        return archs[0]

    # Fall back to model_type
    model_type = config.get("model_type", "")
    if model_type:
        return model_type

    return None


def get_modifiers_from_recipe(
    name: str, model_args: ModelArguments
) -> Tuple[List["Modifier"], dict]:
    # Callers may pass a minimal args object without output_format (data-free recipe
    # routing); None is a no-op for the GPTQ format validation.
    validate_gptq_output_format(getattr(model_args, "output_format", None))
    from qlean.modifiers import ModifierFactory

    if name != "day0":
        recipe = model_args.recipe
        if os.path.exists(recipe):
            with open(recipe, "r") as file:
                recipe_dict = yaml.safe_load(file)
        else:
            raise ValueError(f"Recipe not a valid file_path {recipe}")

        if not isinstance(recipe_dict, dict):
            raise ValueError(f"Cannot parse recipe {recipe}")

        if not ModifierFactory._loaded:
            ModifierFactory.load_from_package()
        
        _try_inject_verify(recipe_dict, model_args)
        modifiers = get_modifiers_from_dict(recipe_dict)
        meta = {
            "input_format": "user_specified",
            "output_format": "user_specified",
            "recipe_path": str(recipe),
            "recipe_content": recipe_dict,
            "modifier_names": [type(m).__name__ for m in modifiers],
            "scheme": None,
        }
        return modifiers, meta
    else:
        # User-specified recipe → run as-is, no route-table replacement
        if model_args.recipe is not None:
            recipe = Path(model_args.recipe)
            logger.info(f"Use input recipe: {recipe}")

            if not recipe.is_file():
                raise ValueError(f"Recipe not found: {recipe}")

            with recipe.open("r") as file:
                recipe_dict = yaml.safe_load(file)
                logger.debug(f"content in yaml: {str(recipe_dict)}")

            if not isinstance(recipe_dict, dict):
                raise ValueError(f"Cannot parse recipe {recipe}")

            if not ModifierFactory._loaded:
                ModifierFactory.load_from_package()
            
            _try_inject_verify(recipe_dict, model_args)
            modifiers = get_modifiers_from_dict(recipe_dict)
            meta = {
                "input_format": "user_specified",
                "output_format": "user_specified",
                "recipe_path": str(recipe),
                "recipe_content": recipe_dict,
                "modifier_names": [type(m).__name__ for m in modifiers],
                "scheme": None,
            }
            return modifiers, meta

        # Auto mode: input/output format required
        output_format = model_args.output_format
        input_format = model_args.input_format or detect_input_format(model_args.model_path)
        logger.info(f"Detected input format: {input_format}, target output format: {output_format}")
        recipe = resolve_recipe(output_format, model_args.model_name, model_args.model_path)
        if recipe is None:
            from qlean.core.auto_recipe import AUTO_SCHEMES, build_auto_recipe

            if model_args.mix_path is not None:
                raise ValueError(
                    "--mix_path requires a model-specific mixed-precision recipe; "
                    "automatic GeneralQuantEngineModifier recipes produce only --save_path"
                )
            recipe_dict = build_auto_recipe(
                input_format, output_format, model_args.model_path,
                architecture=_get_architecture(model_args.model_path),
            )
            _try_inject_verify(recipe_dict, model_args)
            if not ModifierFactory._loaded:
                ModifierFactory.load_from_package()
            modifiers = get_modifiers_from_dict(recipe_dict)
            return modifiers, {
                "input_format": input_format,
                "output_format": output_format,
                "recipe_path": "generated:GeneralQuantEngineModifier",
                "recipe_content": recipe_dict,
                "modifier_names": [type(m).__name__ for m in modifiers],
                "scheme": AUTO_SCHEMES[output_format],
            }

        logger.info(f"Using recipe: {recipe}")

        short_name = model_args.model_name.split("/")[-1]
        is_exact_match = recipe.name == f"{short_name}.yaml"

        route_key = (input_format, output_format)

        if route_key not in MODIFIER_REPLACEMENT and not is_exact_match:
            supported = list(MODIFIER_REPLACEMENT.keys())
            raise ValueError(
                f"Unsupported conversion: {input_format} → {output_format}\n"
                f"Supported combinations: {supported}\n\n"
                f"Note: Matched fallback recipe for '{model_args.model_name}'. "
                f"If a model-specific recipe exists, ensure --model_name "
                f"exactly matches the recipe filename."
            )

        if not recipe.is_file():
            raise ValueError(f"Recipe not found: {recipe}")

        with recipe.open("r") as file:
            recipe_dict = yaml.safe_load(file)
            logger.debug(f"content in yaml: {str(recipe_dict)}")

        if not isinstance(recipe_dict, dict):
            raise ValueError(f"Cannot parse recipe {recipe}")

        gptq_preset = apply_gptq_output_format(recipe_dict, output_format)
        # Apply the scheme override
        scheme_value = gptq_preset or SCHEME_MAP.get(output_format)
        if gptq_preset is not None:
            logger.info(f"Applied GPTQ preset: {gptq_preset}")
        elif scheme_value:
            update_scheme(recipe_dict, scheme_value)
            logger.info(f"Applied scheme: {scheme_value}")
        else:
            logger.info("Using default scheme from recipe")

        if output_format == "BF16":
            _retarget_recipe_to_bf16(recipe_dict, input_format, recipe)

        if not ModifierFactory._loaded:
            ModifierFactory.load_from_package()

        _try_inject_verify(recipe_dict, model_args)
        if route_key in MODIFIER_REPLACEMENT:
            # In the public route table: replace generalDay0Modifier
            modifiers = get_modifiers_from_dict_day0(input_format, output_format, recipe_dict)
        else:
            # Not in the route table, but an exact recipe matched → parse directly
            logger.warning(
                f"Exact recipe match for '{short_name}' — skipping route validation; "
                f"--output-format may be ignored by this recipe. "
                f"Ensure --model_path points to the correct model."
            )
            modifiers = get_modifiers_from_dict(recipe_dict)
        meta = {
            "input_format": input_format,
            "output_format": output_format,
            "recipe_path": str(recipe),
            "recipe_content": recipe_dict,
            "modifier_names": [type(m).__name__ for m in modifiers],
            "scheme": scheme_value,
        }
        return modifiers, meta


def get_modifiers_from_dict(recipe_dict: dict) -> List[Modifier]:
    modifiers = []
    for stage_key, stage_val in recipe_dict.items():
        if stage_key.endswith("_stage") and isinstance(stage_val, dict):
            stage = stage_key.replace("_stage", "")
            for group_key, group_val in stage_val.items():
                if group_key.endswith("_modifiers") and isinstance(group_val, dict):
                    group = group_key.replace("_modifiers", "")
                    for mod_type, mod_args in group_val.items():
                        if mod_args is None:
                            mod_args = {}
                        logger.debug(
                            f"recipe: stage {stage}; group {group}; modifier of type {mod_type}: {mod_args}"
                        )
                        modifier = ModifierFactory.create(
                            mod_type, **mod_args, stage=stage, group=group
                        )
                        modifiers.append(modifier)
    logger.debug(modifiers)
    return modifiers


def get_modifiers_from_dict_day0(
    input_format: str, output_format: str, recipe_dict: dict
) -> List[Modifier]:
    validate_gptq_output_format(output_format)
    route_key = (input_format, output_format)
    replacement = MODIFIER_REPLACEMENT[route_key]

    modifiers = []
    for stage_key, stage_val in recipe_dict.items():
        if stage_key.endswith("_stage") and isinstance(stage_val, dict):
            stage = stage_key.replace("_stage", "")
            for group_key, group_val in stage_val.items():
                if group_key.endswith("_modifiers") and isinstance(group_val, dict):
                    group = group_key.replace("_modifiers", "")
                    for mod_type, mod_args in group_val.items():
                        if mod_args is None:
                            mod_args = {}
                        logger.debug(
                            f"recipe: stage {stage}; group {group}; modifier of type {mod_type}: {mod_args}"
                        )
                        if mod_type == "generalDay0Modifier":
                            # Replace per the route table
                            for new_mod_type in replacement:
                                logger.info(
                                    f"Route: {input_format} → {output_format}, "
                                    f"creating {new_mod_type}"
                                )
                                mod = ModifierFactory.create(
                                    new_mod_type,
                                    **mod_args,
                                    stage=stage,
                                    group=group,
                                )
                                modifiers.append(mod)
                        else:
                            modifier = ModifierFactory.create(
                                mod_type, **mod_args, stage=stage, group=group
                            )
                            modifiers.append(modifier)
    logger.debug(modifiers)
    return modifiers

def remove_unsupported_fields(save_path: str):

    to_remove_names = {"scale_dtype", "zp_dtype"}

    def recursive_remove(obj):
        if isinstance(obj, dict):
            return {k: recursive_remove(v) for k, v in obj.items() if k not in to_remove_names}
        elif isinstance(obj, list):
            return [recursive_remove(item) for item in obj]
        return obj

    import json
    config_file = os.path.join(save_path, "config.json")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config = recursive_remove(config)

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_qlean_metadata(
    save_path: str,
    model_args,
    input_format: str,
    output_format: str,
    recipe_path: str,
    recipe_content: dict,
    modifier_names: List[str],
    scheme: str | None = None,
):
    """Write qlean_config.json, the reproducibility metadata, into the output directory."""
    import datetime

    metadata = {
        "qlean_version": _get_version(),
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": model_args.model_name,
        "model_path": model_args.model_path,
        "input_format": input_format,
        "output_format": output_format,
        "recipe_path": str(recipe_path),
        "recipe_content": recipe_content,
        "modifier_chain": modifier_names,
        "scheme": scheme,
        "cli_args": sys.argv[1:],
    }

    output_file = os.path.join(save_path, "qlean_config.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved quantization metadata to {output_file}")


def _get_version() -> str:
    """Return the installed qlean version."""
    try:
        from importlib.metadata import version
        return version("qlean")
    except Exception:
        return "unknown"
