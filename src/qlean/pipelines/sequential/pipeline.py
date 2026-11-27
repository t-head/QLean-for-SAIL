# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Adapted from llm-compressor's SequentialPipeline

import contextlib
import inspect
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from qlean.args import OtherArguments

import torch
from compressed_tensors.offload import disable_offloading, set_onload_device
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.quantization import disable_quantization, enable_quantization
from loguru import logger

from qlean.core import get_compressor
from qlean.modifiers.gptq.diagnostics import (
    FirstLayerInputDiagnostics,
    diagnostics_enabled,
)
from qlean.utils.dev import get_main_device
from qlean.utils.hooks import HooksMixin

__all__ = ["run_pipeline"]

# Matches decoder layer names such as "model.layers.0" or "transformer.h.11"
_LAYER_PATTERNS = (
    re.compile(r"^(?P<prefix>.*)\.layers\.(?P<index>\d+)$"),
    re.compile(r"^(?P<prefix>.*)\.h\.(?P<index>\d+)$"),
)


class _StopForward(Exception):
    """Raised by the capture hook to abort the full-model forward."""


def run_pipeline(model: Any, dataloader: Any, args: "OtherArguments"):
    """
    Sequential pipeline for layer-wise quantization (e.g. GPTQ).

    Event sequence:
        1. initialize() - Setup modifiers
        2. calibration_epoch_start() - Register hooks
        3. [Capture inputs to the first decoder layer for every batch]
        4. Per decoder layer, in order:
            a. [Forward layer over cached inputs] - Accumulate statistics
            b. sequential_epoch_end() - Quantize this layer
            c. If enabled, [forward quantized layer, hooks off] to propagate error;
               otherwise cache the calibration pass outputs for the next layer
        5. [Calibrate modules after the last layer (final norm, lm_head)]
           sequential_epoch_end() - Quantize them
        6. calibration_epoch_end() - Remove hooks, freeze quantization
        7. finalize() - Final cleanup
    """
    logger.info("Running pipeline: sequential")
    if dataloader is None:
        raise ValueError("GPTQ sequential calibration requires a dataloader")

    compressor = get_compressor()
    try:
        _run_pipeline(model, dataloader, args, compressor)
    except BaseException:
        # Normal finalization would freeze incomplete qparams and mask the failure.
        from qlean.modifiers.gptq import GPTQModifier
        for modifier in compressor.modifiers:
            if isinstance(modifier, GPTQModifier):
                try:
                    modifier._cleanup_calibration(_underlying_model(model))
                except BaseException:
                    logger.exception("Failed to clean up GPTQ calibration resources")
        raise


def _run_pipeline(model, dataloader, args, compressor):
    underlying = _underlying_model(model)
    propagate_error = getattr(args, "propagate_error", True)

    # Step 1-2: initialize modifiers and register calibration hooks
    compressor.initialize()
    set_onload_device(underlying, get_main_device())
    compressor.calibration_epoch_start(sequential=True)

    # Step 3: split the model into decoder layers
    layers, layer_names = _find_decoder_layers(underlying)
    input_diagnostics = (
        FirstLayerInputDiagnostics(layer_names[0]) if diagnostics_enabled() else None
    )
    logger.info(
        f"Sequential pipeline: {len(layers)} decoder layers "
        f"('{layer_names[0]}' ... '{layer_names[-1]}')"
    )
    logger.info(
        "Sequential pipeline: Hessian memory is bounded by a single decoder "
        "layer, so the full-model estimate logged earlier does not apply"
    )
    logger.info(
        "Sequential pipeline: quantization error propagation is {}",
        "enabled" if propagate_error else "disabled",
    )
    outside_targets = _find_outside_targets(underlying, layers)
    if outside_targets:
        logger.info(
            f"{len(outside_targets)} quantization target(s) outside the decoder "
            f"layers ({', '.join(outside_targets[:5])}): they will be calibrated "
            "after the last layer"
        )

    num_calibration_rounds = getattr(args, "num_calibration_rounds", 1)

    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        stack.enter_context(_disable_quantization(underlying))
        _assert_quantization_disabled(underlying, "calibration start")

        # Step 4: ensure calibration batches will not resize offloaded RoPE
        # cos/sin caches. llm-compressor's sequential pipeline expects fixed
        # max_length padding; if the dataloader has variable lengths, we warn
        # and pre-warm the caches to the global maximum observed length.
        max_seq_len = _check_calibration_sequence_length(dataloader)
        if max_seq_len is not None:
            _prewarm_rotary_caches(underlying, max_seq_len)

        # Step 5: capture the inputs to the first layer for every batch
        captured = _capture_first_layer_inputs(
            underlying,
            layers[0],
            dataloader,
            input_diagnostics=input_diagnostics,
        )
        num_batches = len(captured)

        # Step 5: process each layer in order
        for layer_idx, (layer, layer_name) in enumerate(zip(layers, layer_names)):
            desc = f"layer {layer_idx + 1}/{len(layers)} ({layer_name})"
            has_targets = any(
                hasattr(module, "quantization_scheme") for module in layer.modules()
            )

            # Keep only this layer's weights onloaded across every calibration and
            # propagation batch. Exiting the scope releases them before the next
            # layer, preserving sequential pipeline memory bounds.
            with disable_offloading():
                if has_targets:
                    # Every calibration round starts from the same layer input.
                    layer_inputs = captured
                    next_captured = None
                    calib_time = 0.0
                    quant_time = 0.0
                    round_modules = list(layer.modules())
                    for round_idx in range(num_calibration_rounds):
                        is_last_calibration_round = round_idx + 1 == num_calibration_rounds
                        compressor.sequential_epoch_start(
                            modules=round_modules,
                            calibration_scope=layer_name,
                            calibration_round_index=round_idx,
                            num_calibration_rounds=num_calibration_rounds,
                            is_last_calibration_round=is_last_calibration_round,
                        )
                        start = time.time()
                        round_outputs = (
                            [] if not propagate_error and round_idx == 0 else None
                        )

                        for batch_idx, (layer_args, layer_kwargs) in enumerate(
                            layer_inputs
                        ):
                            if layer_idx == 0 and batch_idx == 0 and input_diagnostics:
                                input_diagnostics.observe_before_use(
                                    layer, layer_args, layer_kwargs
                                )
                            outputs = layer(*layer_args, **layer_kwargs)
                            if round_outputs is not None:
                                hidden, residual = _extract_layer_state(layer, outputs)
                                topk_indices = _extract_prev_topk_indices(layer, outputs)
                                next_args, next_kwargs = _replace_hidden_states(
                                    layer_args, layer_kwargs, hidden, residual
                                )
                                round_outputs.append(
                                    _replace_prev_topk_indices(
                                        layer,
                                        next_args,
                                        next_kwargs,
                                        topk_indices,
                                    )
                                )
                            if (
                                batch_idx == 0
                                or batch_idx + 1 == num_batches
                                or (batch_idx + 1) % 10 == 0
                            ):
                                logger.info(
                                    f"Sequential {desc}: calibration round "
                                    f"{round_idx + 1}/{num_calibration_rounds}, batch "
                                    f"{batch_idx + 1}/{num_batches} "
                                    f"({time.time() - start:.1f}s elapsed)"
                                )
                        calib_time += time.time() - start
                        if round_outputs is not None:
                            next_captured = round_outputs

                        start = time.time()
                        compressor.sequential_epoch_end(
                            modules=round_modules,
                            calibration_scope=layer_name,
                            calibration_round_index=round_idx,
                            num_calibration_rounds=num_calibration_rounds,
                            is_last_calibration_round=is_last_calibration_round,
                        )
                        _assert_quantization_disabled(underlying, f"after {layer_name}")
                        quant_time += time.time() - start

                    if not propagate_error:
                        captured = next_captured
                else:
                    # nothing to quantize here; only the forward below is needed
                    # to produce the next layer's inputs
                    calib_time = 0.0
                    quant_time = 0.0
                    logger.info(f"Sequential {desc}: no quantization targets")

                prop_time = 0.0
                if propagate_error or not has_targets:
                    # Re-forward quantized layers only when propagating their error.
                    # Layers without targets always need one forward to advance state.
                    start = time.time()
                    with HooksMixin.disable_hooks():
                        for batch_idx in range(num_batches):
                            layer_args, layer_kwargs = captured[batch_idx]
                            outputs = layer(*layer_args, **layer_kwargs)
                            hidden, residual = _extract_layer_state(layer, outputs)
                            topk_indices = _extract_prev_topk_indices(layer, outputs)
                            next_args, next_kwargs = _replace_hidden_states(
                                layer_args, layer_kwargs, hidden, residual
                            )
                            captured[batch_idx] = _replace_prev_topk_indices(
                                layer,
                                next_args,
                                next_kwargs,
                                topk_indices,
                            )
                            if (
                                batch_idx == 0
                                or batch_idx + 1 == num_batches
                                or (batch_idx + 1) % 10 == 0
                            ):
                                logger.info(
                                    f"Sequential {desc}: propagating batch "
                                    f"{batch_idx + 1}/{num_batches} "
                                    f"({time.time() - start:.1f}s elapsed)"
                                )
                    prop_time = time.time() - start

            logger.info(
                f"Sequential {desc}: {num_batches} batches calibrated in "
                f"{calib_time:.1f}s, quantized in {quant_time:.1f}s, "
                f"propagated in {prop_time:.1f}s"
            )

        # Step 6: modules after the last layer (final norm, lm_head)
        if outside_targets:
            _calibrate_tail(
                underlying,
                captured,
                compressor,
                outside_targets,
                num_calibration_rounds,
            )
            _assert_quantization_disabled(underlying, "after tail calibration")

        # Step 7: remove hooks, freeze and enable quantization while still inside
        # the same calibration scope used by llm-compressor.
        compressor.calibration_epoch_end()

    # Step 8: final cleanup
    compressor.finalize()
    logger.info("Sequential pipeline completed")


@contextlib.contextmanager
def _disable_quantization(module: torch.nn.Module):
    try:
        module.apply(disable_quantization)
        yield
    finally:
        module.apply(enable_quantization)


def _assert_quantization_disabled(model: torch.nn.Module, phase: str):
    enabled = [
        name or "<root>"
        for name, module in model.named_modules()
        if getattr(module, "quantization_enabled", False)
    ]
    if enabled:
        preview = ", ".join(enabled[:10])
        suffix = " ..." if len(enabled) > 10 else ""
        raise RuntimeError(
            "Sequential calibration unexpectedly enabled fake quantization "
            f"during {phase}: {preview}{suffix}"
        )


def _underlying_model(model: Any) -> torch.nn.Module:
    """Unwrap the qlean Model wrapper to the underlying torch module."""
    underlying = getattr(model, "model_", None)
    return model if underlying is None else underlying


def _num_batches(dataloader: Any) -> Optional[int]:
    try:
        return len(dataloader)
    except (TypeError, NotImplementedError):
        return None


def _find_decoder_layers(
    model: torch.nn.Module,
) -> Tuple[List[torch.nn.Module], List[str]]:
    """
    Find the decoder layers of the model.

    Layers are modules whose name matches ``<prefix>.layers.<index>`` (the
    convention used by most HuggingFace decoder models), falling back to
    ``<prefix>.h.<index>`` (GPT-2 style). If several naming groups are found,
    the largest one is used.
    """
    for pattern in _LAYER_PATTERNS:
        groups: Dict[str, Dict[int, Tuple[str, torch.nn.Module]]] = {}
        for name, module in model.named_modules():
            match = pattern.match(name)
            if match is not None:
                groups.setdefault(match.group("prefix"), {})[
                    int(match.group("index"))
                ] = (name, module)

        if not groups:
            continue

        prefix = max(groups, key=lambda p: len(groups[p]))
        if len(groups) > 1:
            logger.warning(
                f"Multiple decoder layer groups found ({sorted(groups)}); "
                f"using '{prefix}' with {len(groups[prefix])} layers"
            )

        layers_by_index = groups[prefix]
        indices = sorted(layers_by_index)
        missing = [i for i in range(indices[-1] + 1) if i not in layers_by_index]
        if missing:
            logger.warning(f"Non-contiguous layer indices: {missing[:10]}")

        return (
            [layers_by_index[i][1] for i in indices],
            [layers_by_index[i][0] for i in indices],
        )

    raise ValueError(
        "Sequential pipeline could not find any decoder layers (no modules "
        "match '<prefix>.layers.<index>'). Use the 'calibration' pipeline "
        "instead."
    )


def _find_outside_targets(
    model: torch.nn.Module, layers: List[torch.nn.Module]
) -> List[str]:
    """Find quantization targets that are not part of any decoder layer."""
    inside = set()
    for layer in layers:
        for module in layer.modules():
            inside.add(id(module))

    names = []
    for name, module in model.named_modules():
        if hasattr(module, "quantization_scheme") and id(module) not in inside:
            names.append(name)
    return names


def _capture_first_layer_inputs(
    model: torch.nn.Module,
    first_layer: torch.nn.Module,
    dataloader: Any,
    input_diagnostics: Optional[FirstLayerInputDiagnostics] = None,
) -> List[Tuple[tuple, dict]]:
    """
    Capture the (args, kwargs) passed to the first decoder layer for every
    calibration batch.

    The full model is called once per batch, but a forward pre-hook on the
    first layer aborts execution right after the embeddings have been
    computed, so capture costs one embedding forward per batch.
    """
    captured: List[Tuple[tuple, dict]] = []

    def capture_hook(module, args, kwargs):
        if input_diagnostics is not None:
            input_diagnostics.observe_capture(module, args, kwargs)
        captured.append((tuple(args), dict(kwargs)))
        raise _StopForward()

    handle = first_layer.register_forward_pre_hook(capture_hook, with_kwargs=True)
    total = _num_batches(dataloader)
    start = time.time()
    try:
        for batch_idx, batch in enumerate(dataloader):
            batch = _to_device(batch, model)
            if isinstance(batch, dict):
                # avoid growing caches; layers must not produce key_values
                batch = dict(batch)
                batch["use_cache"] = False
                try:
                    model(**batch)
                except _StopForward:
                    pass
            else:
                try:
                    model(batch, use_cache=False)
                except _StopForward:
                    pass

            if (
                batch_idx == 0
                or (total is not None and batch_idx + 1 == total)
                or (batch_idx + 1) % 10 == 0
            ):
                suffix = f"/{total}" if total is not None else ""
                logger.info(
                    f"Captured layer inputs for batch {batch_idx + 1}{suffix}"
                )
    finally:
        handle.remove()

    if not captured:
        raise ValueError(
            "Sequential pipeline failed to capture any calibration batches; "
            "no batch reached the first decoder layer"
        )

    logger.info(
        f"Captured inputs for {len(captured)} batches in {time.time() - start:.1f}s"
    )
    return captured


def _calibrate_tail(
    model: torch.nn.Module,
    captured: List[Tuple[tuple, dict]],
    compressor: Any,
    outside_targets: List[str],
    calibrate_round: int = 1,
):
    """
    Calibrate the modules that come after the last decoder layer (the final
    norm and lm_head of standard decoder-only models) using the propagated
    activations, then fire a final compression event for them.
    """
    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
    if not isinstance(norm, torch.nn.Module):
        norm = getattr(base, "ln_f", None)
    if not isinstance(norm, torch.nn.Module):
        # Multimodal composites (Qwen3.5/3.8 ForConditionalGeneration) nest the
        # text model, and its final norm, under `language_model`.
        language_model = getattr(base, "language_model", None)
        norm = getattr(language_model, "norm", None)
    if not isinstance(norm, torch.nn.Module):
        transformer = getattr(model, "transformer", None)
        norm = getattr(transformer, "ln_f", None)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        lm_head = getattr(base, "lm_head", None)

    has_norm = isinstance(norm, torch.nn.Module)
    has_lm_head = isinstance(lm_head, torch.nn.Module)
    if not has_norm and not has_lm_head:
        raise RuntimeError(
            "Sequential pipeline found quantization targets outside decoder "
            "layers but could not calibrate a final norm or lm_head. Add those "
            "targets to ignore or use the calibration pipeline."
        )

    if has_lm_head and not has_norm:
        logger.warning(
            "Sequential pipeline found an lm_head but no final norm; the lm_head "
            "would be calibrated on un-normalized decoder output, giving wrong "
            "quantization parameters. Check the norm attribute name or add "
            "lm_head to ignore."
        )

    # The tail forward runs hooks on descendants too, so they are handled as well.
    handled_ids = {
        id(sub)
        for tail in (norm, lm_head)
        if isinstance(tail, torch.nn.Module)
        for sub in tail.modules()
    }
    handled_targets = {
        name for name, module in model.named_modules() if id(module) in handled_ids
    }
    unhandled_targets = sorted(set(outside_targets) - handled_targets)
    if unhandled_targets:
        raise RuntimeError(
            "Sequential pipeline cannot calibrate quantization targets outside "
            "decoder layers that are not the final norm or lm_head: "
            f"{', '.join(unhandled_targets)}. Add them to ignore or use the "
            "calibration pipeline."
        )

    logger.info("Calibrating modules after the last decoder layer (norm, lm_head)")
    norm_accepts_residual = has_norm and _accepts_residual(norm)
    tail_modules = [
        module
        for module in (norm, lm_head)
        if isinstance(module, torch.nn.Module)
    ]
    with disable_offloading():
        for round_idx in range(calibrate_round):
            is_last_calibration_round = round_idx + 1 == calibrate_round
            compressor.sequential_epoch_start(
                modules=tail_modules,
                calibration_scope="tail",
                calibration_round_index=round_idx,
                num_calibration_rounds=calibrate_round,
                is_last_calibration_round=is_last_calibration_round,
            )
            for layer_args, layer_kwargs in captured:
                hidden = _get_hidden_states(layer_args, layer_kwargs)
                if has_norm:
                    if norm_accepts_residual:
                        residual = _get_residual(layer_args, layer_kwargs)
                        hidden = _extract_hidden_states(norm(hidden, residual))
                    else:
                        hidden = norm(hidden)
                if has_lm_head:
                    lm_head(hidden)

            compressor.sequential_epoch_end(
                modules=tail_modules,
                calibration_scope="tail",
                calibration_round_index=round_idx,
                num_calibration_rounds=calibrate_round,
                is_last_calibration_round=is_last_calibration_round,
            )


def _to_device(batch: Any, model: torch.nn.Module) -> Any:
    device = getattr(model, "device", None)
    if device is None or device.type == "meta":
        return batch
    if isinstance(batch, dict):
        return {
            k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
        }
    # _collate yields a bare tensor (or a tuple/list) for non-dict datasets, and
    # _capture_first_layer_inputs forwards those to the model unchanged.
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, (tuple, list)):
        return type(batch)(v.to(device) if hasattr(v, "to") else v for v in batch)
    return batch


_NO_RESIDUAL = object()
_NO_TOPK_INDICES = object()


def _forward_parameter(
    module: torch.nn.Module, name: str
) -> Optional[inspect.Parameter]:
    try:
        return inspect.signature(module.forward).parameters.get(name)
    except (TypeError, ValueError):
        return None


def _accepts_residual(module: torch.nn.Module) -> bool:
    return _forward_parameter(module, "residual") is not None


def _extract_hidden_states(outputs: Any) -> torch.Tensor:
    """Extract hidden states from a decoder layer output."""
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _extract_layer_state(
    layer: torch.nn.Module, outputs: Any
) -> Tuple[torch.Tensor, Any]:
    if not _accepts_residual(layer):
        return _extract_hidden_states(outputs), _NO_RESIDUAL
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise RuntimeError(
            f"Residual decoder layer {type(layer).__name__} must return "
            "(hidden_states, residual)"
        )
    return outputs[0], outputs[1]


def _extract_prev_topk_indices(layer: torch.nn.Module, outputs: Any) -> Any:
    if _forward_parameter(layer, "prev_topk_indices") is None:
        return _NO_TOPK_INDICES
    # A layer that also accepts `residual` returns
    # (hidden_states, residual, topk_indices); otherwise (hidden_states, topk_indices).
    index = 2 if _accepts_residual(layer) else 1
    if not isinstance(outputs, (tuple, list)) or len(outputs) <= index:
        raise RuntimeError(
            f"DSA decoder layer {type(layer).__name__} must return "
            "(hidden_states, topk_indices) or (hidden_states, residual, topk_indices)"
        )
    return outputs[index]


def _get_hidden_states(args: tuple, kwargs: dict) -> torch.Tensor:
    if args:
        return args[0]
    for key in ("hidden_states", "x"):
        if key in kwargs:
            return kwargs[key]
    raise RuntimeError("Captured decoder input does not contain hidden states")


def _get_residual(args: tuple, kwargs: dict) -> torch.Tensor:
    if len(args) > 1:
        return args[1]
    if "residual" in kwargs:
        return kwargs["residual"]
    raise RuntimeError("Captured residual decoder input does not contain residual")


def _replace_hidden_states(
    args: tuple,
    kwargs: dict,
    hidden_states: torch.Tensor,
    residual: Any = _NO_RESIDUAL,
) -> Tuple[tuple, dict]:
    """Replace the state arguments of a captured decoder layer input."""
    kwargs = dict(kwargs)
    if args:
        args = (hidden_states,) + tuple(args[1:])
    elif "x" in kwargs:
        kwargs["x"] = hidden_states
    else:
        kwargs["hidden_states"] = hidden_states

    if residual is _NO_RESIDUAL:
        return args, kwargs
    if len(args) > 1:
        return (args[0], residual) + args[2:], kwargs
    kwargs["residual"] = residual
    return args, kwargs


def _replace_prev_topk_indices(
    layer: torch.nn.Module,
    args: tuple,
    kwargs: dict,
    topk_indices: Any,
) -> Tuple[tuple, dict]:
    if topk_indices is _NO_TOPK_INDICES:
        return args, kwargs

    parameter = _forward_parameter(layer, "prev_topk_indices")
    if parameter is None:
        return args, kwargs

    kwargs = dict(kwargs)
    if parameter.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        positional_names = [
            name
            for name, candidate in inspect.signature(layer.forward).parameters.items()
            if candidate.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        position = positional_names.index("prev_topk_indices")
        if len(args) > position:
            args = args[:position] + (topk_indices,) + args[position + 1 :]
            kwargs.pop("prev_topk_indices", None)
            return args, kwargs
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise RuntimeError(
                f"DSA decoder layer {type(layer).__name__} omitted positional-only "
                "prev_topk_indices"
            )

    kwargs["prev_topk_indices"] = topk_indices
    return args, kwargs


def _check_calibration_sequence_length(dataloader: Any) -> Optional[int]:
    """Check calibration batch sequence lengths and return the max if variable.

    llm-compressor's sequential pipeline assumes calibration data is padded to a
    fixed ``max_length``. Variable-length batches cause rotary position embedding
    cos/sin caches to be lazily resized, which breaks compressed_tensors'
    shared-CPU offload cache (the underlying shared storage cannot be resized
    in-place and raises ``RuntimeError: setStorage``).

    Scan without a forward, then reduce MIN and MAX unconditionally on every
    distributed rank, including empty/unknown splits. Return the global maximum
    for variable or partly unknown lengths; fixed global lengths need no prewarm.
    """
    try:
        iterator = iter(dataloader) if dataloader is not None else iter(())
    except TypeError:
        logger.debug("Dataloader is not iterable; no local sequence-length bounds")
        iterator = iter((None,))

    min_seq_len = torch.iinfo(torch.long).max
    max_seq_len = 0
    checked = 0
    unknown = False

    for batch in iterator:
        seq_len = _extract_sequence_length(batch)
        if seq_len is None:
            unknown = True
            continue
        min_seq_len = min(min_seq_len, seq_len)
        max_seq_len = max(max_seq_len, seq_len)
        checked += 1

    # Empty/unknown ranks use neutral bounds but must still take part in both
    # collectives, in the same order. Unknown batches request a conservative
    # prewarm to the largest known length on every rank.
    local_min = -1 if unknown else min_seq_len
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        device = get_main_device()
        minimum = torch.tensor(local_min, dtype=torch.long, device=device)
        maximum = torch.tensor(max_seq_len, dtype=torch.long, device=device)
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        local_min, max_seq_len = int(minimum.item()), int(maximum.item())

    if max_seq_len == 0:
        return None
    if local_min != max_seq_len:
        logger.warning(
            f"Calibration batches have variable or unknown sequence lengths "
            f"(global bounds {local_min} ... {max_seq_len}). "
            f"Pre-warming rotary caches to seq_len={max_seq_len} as a fallback. "
            "For best results tokenize with padding='max_length' and a fixed "
            "max_length, as expected by llm-compressor's sequential pipeline."
        )
        return max_seq_len

    logger.info(
        f"Calibration sequence length is consistent ({checked} local batch(es) checked, "
        f"seq_len={max_seq_len})"
    )
    return None


def _extract_sequence_length(batch: Any) -> Optional[int]:
    """Best-effort extraction of the token sequence length from a batch."""
    if isinstance(batch, torch.Tensor):
        tensor = batch
    elif isinstance(batch, (tuple, list)) and batch:
        tensor = batch[0]
    elif isinstance(batch, dict):
        for key in ("input_ids", "inputs_embeds", "input"):
            if key in batch:
                tensor = batch[key]
                if key == "inputs_embeds":
                    if not isinstance(tensor, torch.Tensor) or tensor.dim() < 2:
                        return None
                    return int(tensor.shape[-2])
                break
        else:
            return None
    else:
        return None

    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.dim() >= 2:
        return int(tensor.shape[-1])
    if tensor.dim() == 1:
        return int(tensor.shape[0])
    return None


def _prewarm_rotary_caches(model: torch.nn.Module, seq_len: int):
    """Pre-allocate rotary embedding cos/sin caches to *seq_len*.

    When models are loaded with CPU/disk offloading, the cos/sin tables used by
    rotary position embeddings are offloaded like any other buffer. If batches
    have variable sequence lengths, this helper creates a single cache large
    enough for the longest batch, avoiding repeated resizes that break
    compressed_tensors' shared-CPU offload cache.

    The helper only runs for offloaded models so non-offloaded models are not
    penalized.
    """
    has_offloading = any(
        isinstance(getattr(module, "_buffers", None), OffloadCache)
        for module in model.modules()
    )
    if not has_offloading:
        logger.debug("Model is not offloaded; skipping rotary cache pre-warm")
        return

    device = get_main_device()
    count = 0
    for module in model.modules():
        if not hasattr(module, "_set_cos_sin_cache"):
            continue

        dtype = next(module.buffers(), None)
        dtype = dtype.dtype if dtype is not None else torch.get_default_dtype()

        try:
            module._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            count += 1
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to pre-warm rotary cache for {module.__class__.__name__}"
            )

    if count:
        logger.info(
            f"Pre-warmed {count} rotary embedding cache(s) to seq_len={seq_len}"
        )
