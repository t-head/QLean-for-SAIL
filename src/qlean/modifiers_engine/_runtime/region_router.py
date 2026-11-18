# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import fnmatch
import functools
import logging
from copy import deepcopy
from typing import Callable, Optional, Sequence, Union

from qlean.modifiers_engine._runtime.plan import Plan, ROW_WISE_SCHEMES
from qlean.modifiers_engine._runtime.scale_utils import (
    find_scale_name, output_scale_name, output_weight_name,
)

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_ROWS = 1024 * 1024


def module_name(name: str) -> str:
    if "." not in name:
        return name
    return name.rsplit(".", 1)[0]


def param_name(name: str) -> str:
    # Dotless keys (e.g. multimodal markers such as "image_start") are their own param.
    return name.rsplit(".", 1)[-1]


SRC_KINDS = ("bf16", "fp8_block", "nvfp4", "int4", "mxfp4", "mxfp8")
DST_KINDS = ("bf16", "W8A8", "MXFP4", "FP8_DYNAMIC", "FP8_BLOCK", "acext_int4",
             "passthrough")
PASSTHROUGH_SRCS = ("bf16", "fp8_block", "mxfp4", "mxfp8")
WHEN_KINDS = ("always", "has_source_scale", "no_source_scale")
EXPERT_MODES = ("none", "unpack")
REGION_FIELDS = frozenset({"name", "match", "when", "dst", "expert_mode",
                           "split_gate_up", "transpose", "src",
                           "chunked", "chunk_rows"})
# Fused MoE weights that lack a ".weight" suffix.
_FUSED_EXPERT_PARAMS = frozenset({"gate_up_proj", "down_proj"})

SRC_PROVIDER = {
    "bf16": None,
    "fp8_block": "fp8_block",
    "nvfp4": "nvfp4",
    "int4": "int4",
    "mxfp4": "mxfp4",
    "mxfp8": "mxfp8",
}

SRC_KIND = {"fp8_block": "fp8", "nvfp4": "nvfp4", "int4": "int4",
            "mxfp4": "mxfp4", "mxfp8": "mxfp8"}

CONSUMED_SOURCE_SUFFIXES = {
    "bf16": (),
    "fp8_block": ("_scale_inv", ".scale", "weight_scale"),
    "nvfp4": ("_scale", "_scale_2"),
    "int4": ("weight_scale", "weight_shape"),
    "mxfp4": (".scale", "weight_scale", "weight_offset", "weight_shape", "scale_bias"),
    "mxfp8": (".scale", "weight_scale", "_scale_inv"),
}

_ENGINE_DEFAULT_SCALE_SUFFIXES = ("_scale_inv", ".scale")

# A shared global scalar cannot express conflicting per-source values.
SRC_GLOBAL_PARAMS = {
    "fp8_block": ("dequant_block_size",),
    "nvfp4": ("nvfp4_group_size",),
    "int4": ("int4_group_size",),
}


def normalize_regions(regions: list) -> list:
    if not isinstance(regions, list) or not regions:
        raise ValueError("regions must be a nonempty list")

    normalized = []
    for i, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError(
                f"regions[{i}] must be a dict, got {type(region).__name__}"
            )
        copied = deepcopy(region)
        for field, kinds in (("src", SRC_KINDS), ("dst", DST_KINDS)):
            value = region.get(field)
            if value is None:
                continue
            canonical = {kind.lower(): kind for kind in kinds}
            if not isinstance(value, str) or value.lower() not in canonical:
                raise ValueError(
                    f"regions[{i}] (region {region.get('name')!r}): "
                    f"invalid {field} {value!r}; allowed: {kinds} "
                    f"(case-insensitive)"
                )
            copied[field] = canonical[value.lower()]
        normalized.append(copied)
    return normalized


def _validate_match_pattern(pattern: str) -> None:
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f"match pattern {pattern!r} must be a nonempty glob string")
    if pattern.startswith("re:"):
        raise ValueError(
            f"match pattern {pattern!r}: regex 're:' syntax is not supported; use glob patterns"
        )


def validate_regions(regions: list) -> None:
    if not isinstance(regions, list) or not regions:
        raise ValueError("regions must be a nonempty list")

    seen = set()
    for i, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError(
                f"regions[{i}] must be a dict, got {type(region).__name__}"
            )
        unknown = set(region) - REGION_FIELDS
        if unknown:
            raise ValueError(
                f"regions[{i}] has unknown fields: {sorted(unknown, key=repr)}; "
                f"allowed: {sorted(REGION_FIELDS)}"
            )

        name = region.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"regions[{i}] requires name (nonempty str)")
        if name in seen:
            raise ValueError(
                f"Duplicate region name: {name!r}; hit counts would be merged"
            )
        seen.add(name)

        patterns = region_patterns(region)
        if not patterns:
            raise ValueError(f"region {name!r}: match must be a nonempty list[str]")
        for pattern in patterns:
            try:
                _validate_match_pattern(pattern)
            except ValueError as exc:
                raise ValueError(f"region {name!r}: {exc}") from exc

        when = region.get("when", "always")
        if when not in WHEN_KINDS:
            raise ValueError(
                f"region {name!r}: invalid when {when!r}; allowed: {WHEN_KINDS}"
            )

        dst = region.get("dst")
        if dst is not None and dst not in DST_KINDS:
            raise ValueError(
                f"region {name!r}: invalid dst {dst!r}; allowed: {DST_KINDS} (omit for passthrough)"
            )

        src = region.get("src")
        if src is not None and src not in SRC_KINDS:
            raise ValueError(
                f"region {name!r}: invalid src {src!r}; allowed: {SRC_KINDS}"
            )

        eff_src = effective_src(region)
        if dst is None and eff_src != "bf16":
            raise ValueError(
                f"region {name!r}: missing dst with src={eff_src!r}. "
                f"Without a cast, dequantized output keeps dequant_dtype. "
                f"Set dst: bf16 to dequantize to bf16, "
                f"or dst: passthrough to preserve source weights and scales "
                f"without dequantization or requantization. "
                f"dst is required for config scheme / ignore / expert_dtype"
            )
        if dst == "bf16" and eff_src == "bf16":
            raise ValueError(
                f"region {name!r}: dst: bf16 with src: bf16 is a no-op; "
                f"omit dst for passthrough"
            )

        expert_mode = region.get("expert_mode", "none")
        if expert_mode not in EXPERT_MODES:
            raise ValueError(
                f"region {name!r}: invalid expert_mode {expert_mode!r}; "
                f"allowed: {EXPERT_MODES}"
            )

        split_gate_up = region.get("split_gate_up", False)
        if not isinstance(split_gate_up, bool):
            raise ValueError(
                f"region {name!r}: split_gate_up must be bool, got {split_gate_up!r}"
            )

        transpose = region.get("transpose", "auto")
        if transpose not in ("auto", True, False):
            raise ValueError(
                f"region {name!r}: invalid transpose {transpose!r}; "
                f"allowed: 'auto' | true | false"
            )

        chunked = region.get("chunked", False)
        if not isinstance(chunked, bool):
            raise ValueError(
                f"region {name!r}: chunked must be bool, got {chunked!r}"
            )
        chunk_rows = region.get("chunk_rows")
        if chunk_rows is not None and not (
            isinstance(chunk_rows, int) and not isinstance(chunk_rows, bool)
            and chunk_rows > 0
        ):
            raise ValueError(
                f"region {name!r}: chunk_rows must be a positive integer, got {chunk_rows!r}"
            )
        if not chunked and chunk_rows is not None:
            raise ValueError(
                f"region {name!r}: chunk_rows requires chunked: true"
            )
        if chunked:
            # Streaming requires row-independent source scales and quantization without layout changes.
            if dst not in ROW_WISE_SCHEMES:
                raise ValueError(
                    f"region {name!r}: chunked requires a row-wise dst in "
                    f"{sorted(ROW_WISE_SCHEMES)}, got dst={dst!r}; "
                    f"chunk boundaries can split 128x128 quantization blocks."
                )
            if eff_src != "mxfp8":
                raise ValueError(
                    f"region {name!r}: chunked requires src=mxfp8 (fp8 weights + "
                    f"e8m0 scales grouped by 32 along K, independent along M), "
                    f"got src={eff_src!r}; fp8_block scales also span M, "
                    f"so chunk boundaries can split scale blocks."
                )
            if expert_mode != "none":
                raise ValueError(
                    f"region {name!r}: chunked does not support expert_mode={expert_mode!r} "
                    f"(unpacking changes layout; streaming requires a single 2D tensor)."
                )
            if split_gate_up:
                raise ValueError(
                    f"region {name!r}: chunked does not support split_gate_up."
                )

        if dst == "passthrough":
            if eff_src not in PASSTHROUGH_SRCS:
                raise ValueError(
                    f"region {name!r}: passthrough supports {PASSTHROUGH_SRCS}; "
                    f"src={eff_src!r} requires additional auxiliary-tensor semantics"
                )
            if expert_mode != "none" or split_gate_up or transpose is True:
                raise ValueError(
                    f"region {name!r}: passthrough cannot unpack, split or transpose weights"
                )

    last = regions[-1]
    last_match = last.get("match", "*")
    if last_match != "*" or last.get("when", "always") != "always":
        raise ValueError(
            "regions must end with an explicit fallback "
            "{name: fallback, match: '*', when: always}; "
            "do not rely on implicit copy when nothing matches"
        )


def effective_src(region: dict) -> str:
    # YAML src: null must use the default too.
    return region.get("src") or "bf16"


def is_passthrough(region: dict) -> bool:
    return region.get("dst") == "passthrough"


def spec_lazy_input(name: str, shape: list, cfg: dict, weight_map=None) -> bool:
    if "." not in name:
        return False
    region = match_region(name, shape, weight_map, cfg["_regions"])
    return region is not None and is_passthrough(region)


def validate_region_srcs(regions: list) -> None:
    param_owner = {}
    for region in regions:
        src = effective_src(region)
        for param in SRC_GLOBAL_PARAMS.get(src, ()):
            owner = param_owner.get(param)
            if owner is not None and owner != src:
                raise NotImplementedError(
                    f"src {owner!r} and src {src!r} share global parameter {param!r}; "
                    f"conflicting scalar values require a future per-region "
                    f"src {{kind, params}} schema"
                )
            param_owner[param] = src


def _region_when_ok(name: str, weight_map: Optional[dict],
                    region: dict) -> bool:
    when = region.get("when", "always")
    if when == "always":
        return True

    src = effective_src(region)
    if src == "bf16" or weight_map is None:
        has = False
    else:
        has = find_scale_name(
            name, weight_map, kind=SRC_KIND[src]
        ) is not None

    return has if when == "has_source_scale" else not has


def _match_pattern(name: str, pattern: Union[str, list]) -> bool:
    patterns = pattern if isinstance(pattern, list) else [pattern]
    if not patterns:
        raise ValueError("match must be a nonempty list[str]")
    for item in patterns:
        _validate_match_pattern(item)
    return any(fnmatch.fnmatchcase(name, item) for item in patterns)


def region_patterns(region: dict) -> list:
    match = region.get("match", "*")
    return list(match) if isinstance(match, list) else [match]


def match_patterns(name: str, region: dict) -> list:
    """Ignore when to distinguish pattern misses from condition rejections."""
    return [p for p in region_patterns(region) if _match_pattern(name, p)]


def match_region(name: str, shape: list,
                 weight_map: Optional[dict], regions: list):
    for region in regions:
        if not _match_pattern(name, region.get("match", "*")):
            continue
        if not _region_when_ok(name, weight_map, region):
            continue
        return region
    return None


def _resolve_transpose(name: str, shape: list, region: dict, cfg: dict) -> bool:
    transpose = region.get("transpose", "auto")
    if transpose is True or transpose is False:
        return transpose
    hidden = cfg.get("hidden_size")
    if hidden is None:
        raise ValueError(
            "transpose: auto requires hidden_size, got None; "
            "check recipe hidden_size or config.json "
            "text_config.hidden_size / hidden_size"
        )
    if param_name(name) == "gate_up_proj":
        return hidden == shape[1]
    return hidden == shape[-1]


def _scale_spec(out_base: str, cfg: dict):
    if out_base.endswith("weight_packed"):
        raise ValueError(
            f"_scale_spec received an unnormalized packed output key {out_base!r}; use"
            f" output_weight_name first to produce *.weight "
            f"and avoid unsupported weight_packed/weight_scale pairs"
        )
    if cfg.get("scale_naming", "underscore") == "dot":
        return output_scale_name(out_base, ".scale")
    return None


def compile_region_to_ops(name: str, shape: list, region: dict,
                          cfg: dict) -> list:
    dst = region.get("dst")
    expert_mode = region.get("expert_mode", "none")

    if expert_mode == "unpack" and len(shape) != 3:
        raise ValueError(
            f"region {region.get('name')!r}: expert_mode={expert_mode} "
            f"requires 3D, but {name} is {len(shape)}D "
            f"(shape={list(shape)}); refusing a silent 2D fallback"
        )
    if dst not in (None, "bf16") and len(shape) < 2 and not (
            is_passthrough(region) and effective_src(region) == "bf16"):
        raise ValueError(
            f"region {region.get('name')!r}: dst={dst} requires at least 2D, "
            f"but {name} is {len(shape)}D (shape={list(shape)})"
        )

    if dst is None:
        return [("copy", name)]

    if dst == "bf16":
        return [("cast", output_weight_name(name), "bf16")]

    if is_passthrough(region):
        src = effective_src(region)
        out_base = output_weight_name(name)
        if src == "bf16":
            return [("copy", out_base)]
        return [("passthrough_quant", out_base, src, _scale_spec(out_base, cfg))]

    if expert_mode == "unpack":
        transpose = _resolve_transpose(name, shape, region, cfg)
        split_gate_up = region.get("split_gate_up", False)
        pn = param_name(name)
        mname = module_name(name)
        ops = []
        for i in range(shape[0]):
            if split_gate_up and pn == "gate_up_proj":
                for role, proj in (("gate", "gate_proj"), ("up", "up_proj")):
                    out_base = f"{mname}.{i}.{proj}.weight"
                    ops.append(("expert_quant", out_base, i, role,
                                transpose, dst, _scale_spec(out_base, cfg)))
            else:
                out_base = f"{mname}.{i}.{pn}.weight"
                ops.append(("expert_quant", out_base, i, "full",
                            transpose, dst, _scale_spec(out_base, cfg)))
        return ops

    out_base = output_weight_name(name)
    if dst == "acext_int4":
        return [("quant_pack_acext", out_base, _scale_spec(out_base, cfg))]
    return [("quant", out_base, dst, _scale_spec(out_base, cfg))]


def spec_plan_fn(name: str, shape: list, cfg: dict, weight_map=None) -> Plan:
    if "." not in name:
        return Plan([("copy", name)], tag="TOP")

    meta = {"module_name": module_name(name)}

    region = match_region(name, shape, weight_map, cfg["_regions"])
    if region is None:
        return Plan([("copy", name)], tag="COPY", meta=meta)

    ops = compile_region_to_ops(name, shape, region, cfg)
    if is_passthrough(region) and effective_src(region) != "bf16" and weight_map is not None:
        if find_scale_name(name, weight_map, kind=SRC_KIND[effective_src(region)]) is None:
            raise ValueError(
                f"[passthrough] {name}: source scale missing from weight_map; "
                f"use when: has_source_scale to exclude unscaled weights"
            )
    meta["region"] = region["name"]
    meta["dst"] = region.get("dst")
    meta["src"] = effective_src(region)
    return Plan(ops, tag=region["name"].upper(), meta=meta)


def spec_on_plan(name: str, plan: Plan, ctx) -> None:
    # Use region metadata, not tags, to avoid COPY/TOP name collisions.
    meta = plan.meta or {}
    rname = meta.get("region")
    dst = meta.get("dst")
    passthrough = dst == "passthrough"
    kept = (rname is None or dst is None or dst == "bf16"
            or (passthrough and meta.get("src") == "bf16"))

    if rname is not None:
        ctx.bump(f"hit_{rname.lower()}")
    ctx.bump("kept_as_is" if kept or passthrough else "quant_count")

    # Only kept weights belong in config.ignore.
    if kept and (any(op[1].endswith(".weight") for op in plan.ops)
                 or param_name(name) in _FUSED_EXPERT_PARAMS):
        ctx.append("bf16_unquantized", meta.get("module_name"))

    if passthrough:
        ctx.bump("passthrough_count")

    for op in plan.ops:
        if op[0] == "quant" and op[2] == "FP8_DYNAMIC":
            ctx.append("fp8_channelwise_found", meta.get("module_name"))
            break
        if op[0] == "expert_quant" and op[5] == "FP8_DYNAMIC":
            ctx.append("fp8_channelwise_found", meta.get("module_name"))
            break
        if ((op[0] == "quant" and op[2] == "FP8_BLOCK")
                or (op[0] == "expert_quant" and op[5] == "FP8_BLOCK")
                or (op[0] == "passthrough_quant" and op[2] == "fp8_block")):
            # Broad region patterns may include experts handled by an earlier region.
            ctx.append("fp8_blockwise_found", module_name(op[1]))


def derive_dequant_map(regions: list = ()):
    """Keep None entries to preserve first-match masking and packed passthrough inputs."""
    dequant_map = {}
    for region in regions:
        match = region.get("match", "*")
        patterns = match if isinstance(match, list) else [match]
        provider = (None if is_passthrough(region)
                    else SRC_PROVIDER[effective_src(region)])
        for pattern in patterns:
            if pattern in dequant_map and dequant_map[pattern] != provider:
                raise ValueError(
                    f"match {pattern!r} has conflicting dequantization providers "
                    f"({dequant_map[pattern]!r} vs {provider!r}; "
                    f"effective src or passthrough differs); dequant_map cannot represent this"
                )
            dequant_map[pattern] = provider
    if not dequant_map or all(v is None for v in dequant_map.values()):
        return None
    return dequant_map


def derive_stream_map(regions: list = ()):
    stream_map = {}
    for region in regions:
        if not region.get("chunked", False):
            continue
        chunk_rows = region.get("chunk_rows", DEFAULT_CHUNK_ROWS)
        match = region.get("match", "*")
        patterns = match if isinstance(match, list) else [match]
        for pattern in patterns:
            if pattern in stream_map and stream_map[pattern] != chunk_rows:
                raise ValueError(
                    f"match {pattern!r} has conflicting chunk_rows "
                    f"({stream_map[pattern]} vs {chunk_rows}); stream_map cannot represent this"
                )
            stream_map[pattern] = chunk_rows
    return stream_map or None


def _spec_filter_source_key(key: str, suffixes: tuple) -> bool:
    return not any(key.endswith(s) for s in suffixes)


def derive_filter_source_key(regions: list = ()):
    suffix_set = set()
    for region in regions:
        suffix_set.update(CONSUMED_SOURCE_SUFFIXES[effective_src(region)])
    if not suffix_set or suffix_set == set(_ENGINE_DEFAULT_SCALE_SUFFIXES):
        return None
    return functools.partial(_spec_filter_source_key,
                             suffixes=tuple(sorted(suffix_set)))


def report_region_hits(regions: list, stats: dict) -> None:
    """Use global hits to catch bad patterns without false misses from expert-only shards."""
    zero = []
    last = len(regions) - 1
    for i, region in enumerate(regions):
        rname = region["name"]
        count = stats.get(f"hit_{rname.lower()}", 0)
        if i == last and count == 0:
            logger.info(
                f"[region hits] {rname}: 0 (fallback unused; earlier regions "
                f"covered all tensors, as expected)"
            )
            continue
        logger.info(f"[region hits] {rname}: {count}")
        if count == 0:
            zero.append(rname)
    if zero:
        raise ValueError(
            f"Regions with 0 hits (check match patterns; tensors may have been "
            f"silently copied by the fallback): {zero}"
        )
