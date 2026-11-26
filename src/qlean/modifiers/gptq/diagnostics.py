# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import inspect
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

DIAGNOSTICS_ENV = "QLEAN_GPTQ_DIAGNOSTICS"
WEIGHT_SAMPLE_SIZE = 48
SKIP_WEIGHT_AUDIT_LIMIT = 128
EXTREME_INPUT_THRESHOLD = 1e10
EXTREME_WEIGHT_THRESHOLD = 1e4
ROTARY_UNIT_TOLERANCE = 1e-4
GPU_HEADROOM_WARN_BYTES = 2 * 1024**3
_MEM_REFRESH_SECONDS = 0.25
_LAYER_RE = re.compile(
    r"^(?:(?P<prefix>.*)\.)?(?:layers|h)\.(?P<index>\d+)$"
)


def diagnostics_enabled() -> bool:
    value = os.environ.get(DIAGNOSTICS_ENV, "0").strip().lower() or "0"
    return value not in {"0", "false", "no", "off"}


def rank_prefix() -> str:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return f"[rank {torch.distributed.get_rank()} GPTQ diag]"
    return "[rank 0 GPTQ diag]"


def _gpu_free_bytes() -> Optional[int]:
    """Free bytes of the current accelerator device, None when unavailable."""
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.mem_get_info()[0])
    except Exception:
        return None


def _disable_onloading():
    """Read offloaded tensors in place, without moving them to a device.

    In compressed-tensors, accessing an offloaded parameter onloads it to the
    execution device (disk reads go straight to gpu), so any inspection which
    only needs metadata (dtype, shape) must go through this context.
    """
    try:
        from compressed_tensors.offload import disable_onloading
    except Exception:
        return contextlib.nullcontext()
    return disable_onloading()


@contextlib.contextmanager
def _materialize_on_cpu(module: nn.Module):
    """Read a disk-offloaded module's tensors on cpu, never on gpu."""
    try:
        from compressed_tensors.offload import align_module_device
    except Exception:
        yield
        return
    with align_module_device(module, torch.device("cpu")):
        yield


@dataclass
class TensorScan:
    finite: bool = True
    nan: int = 0
    inf: int = 0
    minimum: float = float("nan")
    maximum: float = float("nan")
    absmax: float = float("nan")
    numel: int = 0
    dtype: str = ""
    device: str = ""
    sample: List[float] = field(default_factory=list)


def scan_tensor(tensor: Optional[torch.Tensor], sample_limit: int = 5) -> TensorScan:
    """Finiteness, range and NaN/Inf counts of the finite part of *tensor*."""
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return TensorScan()
    if not tensor.is_floating_point():
        return TensorScan(
            numel=int(tensor.numel()),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
        )

    with torch.no_grad():
        finite_mask = torch.isfinite(tensor)
        scan = TensorScan(
            numel=int(tensor.numel()),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
        )
        finite_values = tensor if bool(finite_mask.all()) else tensor[finite_mask]
        if finite_values.numel() > 0:
            minimum, maximum = torch.aminmax(finite_values.detach())
            scan.minimum = float(minimum)
            scan.maximum = float(maximum)
            scan.absmax = max(abs(scan.minimum), abs(scan.maximum))
        if scan.numel == finite_values.numel():
            return scan

        scan.finite = False
        scan.nan = int(torch.isnan(tensor).sum())
        scan.inf = int(torch.isinf(tensor).sum())
        scan.sample = [
            float(value)
            for value in tensor[~finite_mask].flatten()[:sample_limit].tolist()
        ]
        return scan


@dataclass
class InputRecord:
    name: str = ""
    calls: int = 0
    # index of the first non-finite input within the global hook sequence
    first_nonfinite_seq: int = -1
    first_nonfinite_call: int = -1
    nonfinite_calls: int = 0
    nan: int = 0
    inf: int = 0
    absmax: float = float("nan")
    sample: List[float] = field(default_factory=list)
    dtype: str = ""
    device: str = ""
    shape: Tuple[int, ...] = ()
    extreme_reported: bool = False


@dataclass
class AttentionRecord:
    sequence: int
    shape: Tuple[int, ...]
    scan: Optional[TensorScan]


@dataclass
class RotaryInputScan:
    shape: Tuple[int, ...]
    dtype: str
    device: str
    is_complex: bool
    real_nan: int
    real_inf: int
    imag_nan: int
    imag_inf: int
    magnitude_min: float
    magnitude_max: float
    max_unit_error: float
    storage_ptr: int
    storage_offset: int
    version: int

    @property
    def valid(self) -> bool:
        return (
            self.is_complex
            and not any((self.real_nan, self.real_inf, self.imag_nan, self.imag_inf))
            and self.max_unit_error <= ROTARY_UNIT_TOLERANCE
        )


@dataclass
class MaskInputScan:
    shape: Tuple[int, ...]
    dtype: str
    device: str
    nan: int
    positive_inf: int
    finite_per_row_min: int
    finite_per_row_max: int
    fully_masked_rows: int
    storage_ptr: int
    storage_offset: int
    version: int

    @property
    def valid(self) -> bool:
        return self.nan == 0 and self.positive_inf == 0 and self.fully_masked_rows == 0


@dataclass
class LayerInputSnapshot:
    sequence_length: int
    rotary: Optional[RotaryInputScan]
    mask: Optional[MaskInputScan]
    rotary_skipped: bool = False
    mask_skipped: bool = False


class FirstLayerInputDiagnostics:
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.captured: Optional[LayerInputSnapshot] = None
        self.before_use: Optional[LayerInputSnapshot] = None
        self.verdict: Optional[str] = None
        self._capture_attempted = False
        self._before_use_attempted = False

    def observe_capture(self, module: nn.Module, args: tuple, kwargs: dict):
        if self._capture_attempted:
            return
        self._capture_attempted = True
        try:
            values = self._bind_arguments(module, args, kwargs)
            freqs_cis = values.get("freqs_cis")
            if not isinstance(freqs_cis, torch.Tensor):
                return
            self.captured = self._snapshot(freqs_cis, values.get("mask"))
            self._log_snapshot("capture", self.captured)
        except Exception as exc:
            logger.warning(
                f"{rank_prefix()} first-layer capture argument audit failed: {exc!r}"
            )

    def observe_before_use(self, module: nn.Module, args: tuple, kwargs: dict):
        if self.captured is None or self._before_use_attempted:
            return
        self._before_use_attempted = True
        try:
            values = self._bind_arguments(module, args, kwargs)
            freqs_cis = values.get("freqs_cis")
            if not isinstance(freqs_cis, torch.Tensor):
                return
            self.before_use = self._snapshot(freqs_cis, values.get("mask"))
            self._log_snapshot("before layer-0 use", self.before_use)
            self._report_verdict()
        except Exception as exc:
            logger.warning(
                f"{rank_prefix()} first-layer pre-use argument audit failed: {exc!r}"
            )

    @staticmethod
    def _bind_arguments(module: nn.Module, args: tuple, kwargs: dict) -> Dict[str, Any]:
        try:
            bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
            return dict(bound.arguments)
        except (TypeError, ValueError):
            return dict(kwargs)

    @staticmethod
    def _should_skip_scan(what: str, tensor: torch.Tensor) -> bool:
        needed = tensor.numel() * tensor.element_size() * 3
        free = _gpu_free_bytes() if tensor.is_cuda else None
        if free is None or free >= needed:
            return False
        logger.warning(
            f"{rank_prefix()} skipped {what} for first-layer inputs "
            f"({needed / (1 << 20):.0f} MiB estimated temporary memory, "
            f"{free / (1 << 20):.0f} MiB free on {tensor.device})"
        )
        return True

    @staticmethod
    def _snapshot(freqs_cis: torch.Tensor, mask: Any) -> LayerInputSnapshot:
        rotary_skipped = FirstLayerInputDiagnostics._should_skip_scan(
            "freqs_cis scan", freqs_cis
        )
        mask_skipped = isinstance(
            mask, torch.Tensor
        ) and FirstLayerInputDiagnostics._should_skip_scan("causal mask scan", mask)
        rotary = (
            None
            if rotary_skipped
            else FirstLayerInputDiagnostics._scan_rotary(freqs_cis)
        )
        mask_scan = (
            FirstLayerInputDiagnostics._scan_mask(mask)
            if isinstance(mask, torch.Tensor) and not mask_skipped
            else None
        )
        sequence_length = int(freqs_cis.shape[0]) if freqs_cis.ndim else 0
        return LayerInputSnapshot(
            sequence_length=sequence_length,
            rotary=rotary,
            mask=mask_scan,
            rotary_skipped=rotary_skipped,
            mask_skipped=mask_skipped,
        )

    @staticmethod
    def _storage_signature(tensor: torch.Tensor) -> Tuple[int, int, int]:
        try:
            return (
                int(tensor.untyped_storage().data_ptr()),
                int(tensor.storage_offset()),
                int(tensor._version),
            )
        except Exception:
            return -1, -1, -1

    @staticmethod
    def _scan_rotary(freqs_cis: torch.Tensor) -> RotaryInputScan:
        ptr, offset, version = FirstLayerInputDiagnostics._storage_signature(freqs_cis)
        if not torch.is_complex(freqs_cis):
            return RotaryInputScan(
                shape=tuple(freqs_cis.shape),
                dtype=str(freqs_cis.dtype),
                device=str(freqs_cis.device),
                is_complex=False,
                real_nan=0,
                real_inf=0,
                imag_nan=0,
                imag_inf=0,
                magnitude_min=float("nan"),
                magnitude_max=float("nan"),
                max_unit_error=float("nan"),
                storage_ptr=ptr,
                storage_offset=offset,
                version=version,
            )

        with torch.no_grad():
            real = freqs_cis.real
            imag = freqs_cis.imag
            finite = torch.isfinite(real) & torch.isfinite(imag)
            magnitudes = freqs_cis.abs()
            finite_magnitudes = magnitudes[finite]
            if finite_magnitudes.numel():
                minimum, maximum = torch.aminmax(finite_magnitudes)
                unit_error = (finite_magnitudes - 1).abs().max()
                magnitude_min = float(minimum)
                magnitude_max = float(maximum)
                max_unit_error = float(unit_error)
            else:
                magnitude_min = float("nan")
                magnitude_max = float("nan")
                max_unit_error = float("nan")
            return RotaryInputScan(
                shape=tuple(freqs_cis.shape),
                dtype=str(freqs_cis.dtype),
                device=str(freqs_cis.device),
                is_complex=True,
                real_nan=int(torch.isnan(real).sum()),
                real_inf=int(torch.isinf(real).sum()),
                imag_nan=int(torch.isnan(imag).sum()),
                imag_inf=int(torch.isinf(imag).sum()),
                magnitude_min=magnitude_min,
                magnitude_max=magnitude_max,
                max_unit_error=max_unit_error,
                storage_ptr=ptr,
                storage_offset=offset,
                version=version,
            )

    @staticmethod
    def _scan_mask(mask: torch.Tensor) -> MaskInputScan:
        ptr, offset, version = FirstLayerInputDiagnostics._storage_signature(mask)
        with torch.no_grad():
            finite_per_row = torch.isfinite(mask).sum(dim=-1)
            return MaskInputScan(
                shape=tuple(mask.shape),
                dtype=str(mask.dtype),
                device=str(mask.device),
                nan=int(torch.isnan(mask).sum()),
                positive_inf=int(torch.isposinf(mask).sum()),
                finite_per_row_min=int(finite_per_row.min()),
                finite_per_row_max=int(finite_per_row.max()),
                fully_masked_rows=int((finite_per_row == 0).sum()),
                storage_ptr=ptr,
                storage_offset=offset,
                version=version,
            )

    def _log_snapshot(self, stage: str, snapshot: LayerInputSnapshot):
        prefix = rank_prefix()
        rotary = snapshot.rotary
        if rotary is None:
            logger.warning(
                f"{prefix} {self.layer_name} {stage} freqs_cis: scan skipped for "
                "low VRAM"
            )
        else:
            logger.info(
                f"{prefix} {self.layer_name} {stage} freqs_cis: "
                f"shape={rotary.shape} dtype={rotary.dtype} device={rotary.device} "
                f"complex={rotary.is_complex} real_nan={rotary.real_nan} "
                f"real_inf={rotary.real_inf} "
                f"imag_nan={rotary.imag_nan} imag_inf={rotary.imag_inf} "
                f"magnitude=[{rotary.magnitude_min:.6g}, "
                f"{rotary.magnitude_max:.6g}] "
                f"max_unit_error={rotary.max_unit_error:.3e} "
                f"storage=({rotary.storage_ptr}, {rotary.storage_offset}, "
                f"v{rotary.version})"
            )
        mask = snapshot.mask
        if mask is None:
            detail = "scan skipped for low VRAM" if snapshot.mask_skipped else "None"
            logger.info(f"{prefix} {self.layer_name} {stage} mask: {detail}")
            return
        logger.info(
            f"{prefix} {self.layer_name} {stage} mask: shape={mask.shape} "
            f"dtype={mask.dtype} device={mask.device} nan={mask.nan} "
            f"positive_inf={mask.positive_inf} "
            f"finite_per_row=[{mask.finite_per_row_min}, {mask.finite_per_row_max}] "
            f"fully_masked_rows={mask.fully_masked_rows} "
            f"storage=({mask.storage_ptr}, {mask.storage_offset}, v{mask.version})"
        )

    @staticmethod
    def _mask_valid(snapshot: LayerInputSnapshot) -> bool:
        if snapshot.mask is not None:
            return snapshot.mask.valid
        return snapshot.sequence_length <= 1 and not snapshot.mask_skipped

    @staticmethod
    def _storage_changed(before: Any, after: Any) -> bool:
        if before is None or after is None:
            return False
        return (
            before.storage_ptr,
            before.storage_offset,
            before.version,
        ) != (
            after.storage_ptr,
            after.storage_offset,
            after.version,
        )

    def _report_verdict(self):
        captured = self.captured
        before_use = self.before_use
        if captured is None or before_use is None:
            return

        if captured.rotary is not None and not captured.rotary.valid:
            verdict = "freqs_cis was already invalid when layer inputs were captured"
        elif not captured.mask_skipped and not self._mask_valid(captured):
            verdict = "causal mask was already invalid when layer inputs were captured"
        elif (
            captured.rotary is not None
            and before_use.rotary is not None
            and not before_use.rotary.valid
        ):
            verdict = "freqs_cis became invalid while cached before layer-0 use"
        elif (
            not captured.mask_skipped
            and self._mask_valid(captured)
            and not before_use.mask_skipped
            and not self._mask_valid(before_use)
        ):
            verdict = "causal mask became invalid while cached before layer-0 use"
        elif any(
            (
                captured.rotary_skipped,
                captured.mask_skipped,
                before_use.rotary_skipped,
                before_use.mask_skipped,
            )
        ):
            verdict = (
                "first-layer argument audit was inconclusive because a freqs_cis "
                "or causal mask scan was skipped for low VRAM"
            )
        else:
            changed = []
            if self._storage_changed(captured.rotary, before_use.rotary):
                changed.append("freqs_cis")
            if self._storage_changed(captured.mask, before_use.mask):
                changed.append("causal mask")
            verdict = (
                "freqs_cis and causal mask were valid at capture and before layer-0 "
                f"use; storage identity changed for {', '.join(changed)}"
                if changed
                else (
                    "freqs_cis and causal mask were valid and storage-stable at "
                    "capture and before layer-0 use"
                )
            )

        self.verdict = verdict
        log = logger.info if "were valid" in verdict else logger.error
        log(f"{rank_prefix()} first-layer argument verdict: {verdict}")


class CalibrationDiagnostics:
    """Observational instrumentation wired into :class:`GPTQModifier`."""

    def __init__(self, model: nn.Module, module_names: Dict[nn.Module, str]):
        self.model = model
        self.module_names = dict(module_names)
        self.records: Dict[nn.Module, InputRecord] = {}
        self.baseline: Dict[str, TensorScan] = {}
        self.baseline_checked: Dict[str, bool] = {}
        self.baseline_changed: List[str] = []
        self.skips: Dict[str, str] = {}
        self.site_reports: Dict[str, str] = {}
        self.site_calls: Dict[str, int] = {}
        self.attention_records: Dict[str, AttentionRecord] = {}
        self.attention_verdict: Optional[str] = None
        self.attention_sequence = 0
        self.sequence = 0
        self.batch_index = 0
        self.skip_weight_audits = 0
        self.mem_skips = 0
        self.disabled = False
        self._register_hook: Optional[Callable] = None
        self._mem_free: Optional[int] = None
        self._mem_checked_at = 0.0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self, register_hook: Callable):
        self._register_hook = register_hook
        try:
            self.log_environment()
            self.capture_weight_baseline()
            self.install_probes()
        except Exception as exc:  # diagnostics must never break calibration
            self.disable(f"startup audit failed: {exc!r}")

    def close(self):
        self.records.clear()
        self.baseline.clear()
        self.attention_records.clear()
        self.model = None

    def disable(self, reason: str):
        if not self.disabled:
            self.disabled = True
            logger.warning(f"{rank_prefix()} diagnostics disabled: {reason}")

    # ------------------------------------------------------------------
    # environment
    # ------------------------------------------------------------------

    def log_environment(self):
        prefix = rank_prefix()
        self._report_gpu_memory(prefix)

        devices, offload_dirs = self._device_census()
        categories: Dict[str, int] = {}
        for name in self.module_names.values():
            key = self._category(name)
            categories[key] = categories.get(key, 0) + 1

        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        logger.info(f"{prefix} environment: world_size={world_size}")
        logger.info(
            f"{prefix} environment: {len(self.module_names)} modules to quantize "
            f"({', '.join(f'{k}={v}' for k, v in sorted(categories.items()))})"
        )
        if devices:
            disk = devices.get("disk", 0)
            logger.info(
                f"{prefix} environment: execution devices={devices}"
                + (
                    f"; {disk} modules execute from disk offload under {offload_dirs}"
                    if disk
                    else ""
                )
            )
        elif offload_dirs:
            logger.info(f"{prefix} environment: offload dirs={offload_dirs}")

    @staticmethod
    def _category(name: str) -> str:
        if "shared_experts" in name:
            return "shared_expert"
        if ".experts." in name:
            return "expert_linear"
        if "self_attn" in name or ".attn." in name:
            return "attention"
        if ".mlp." in name:
            return "dense_mlp"
        return "other"

    def _device_census(self) -> Tuple[Dict[str, int], List[str]]:
        counts: Dict[str, int] = {}
        offload_dirs = set()
        for name, module in self.model.named_modules():
            hook = getattr(module, "_hf_hook", None)
            if hook is None:
                continue
            device = getattr(hook, "execution_device", None)
            key = str(device) if device is not None else "unknown"
            if getattr(hook, "offload", None):
                key = "disk"
            counts[key] = counts.get(key, 0) + 1
            if getattr(hook, "offload", None):
                offload_dirs.add(str(hook.offload))
        return counts, sorted(offload_dirs)

    def _report_gpu_memory(self, prefix: str):
        free = _gpu_free_bytes()
        if free is None:
            return
        gib = float(1 << 30)
        device = torch.cuda.current_device()
        total = int(torch.cuda.get_device_properties(device).total_memory)
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        message = (
            f"{prefix} gpu{device} at calibration start: pytorch_allocated="
            f"{allocated / gib:.2f} GiB, reserved={reserved / gib:.2f} GiB, "
            f"free={free / gib:.2f} GiB of {total / gib:.2f} GiB"
        )
        if free < GPU_HEADROOM_WARN_BYTES:
            logger.error(
                f"{message} - almost no free VRAM. The sequential pipeline "
                "onloads one decoder layer at a time and needs several GiB of "
                "headroom per gpu, so expect CUDA OOM during calibration. Check "
                "the load/dispatch: weights should live on cpu/disk "
                "(device_map='auto_offload' inside the compressed-tensors "
                "loader context), not on gpu."
            )
        else:
            logger.info(message)

    def _headroom_ok(self, needed_bytes: int) -> bool:
        """Cached check that the gpu can spare *needed_bytes* for a scan."""
        if needed_bytes <= 0:
            return True
        now = time.monotonic()
        if now - self._mem_checked_at >= _MEM_REFRESH_SECONDS:
            self._mem_checked_at = now
            self._mem_free = _gpu_free_bytes()
        if self._mem_free is None:
            return True
        return self._mem_free >= needed_bytes

    def _should_skip_scan(self, what: str, tensor: torch.Tensor) -> bool:
        """True when scanning *tensor* would risk an OOM on the gpu."""
        needed = tensor.numel() * tensor.element_size() * 3
        if not tensor.is_cuda or self._headroom_ok(needed):
            return False
        self._note_mem_skip(what, tensor)
        return True

    def _note_mem_skip(self, what: str, tensor: torch.Tensor):
        self.mem_skips += 1
        if self.mem_skips <= 3 or self.mem_skips % 64 == 0:
            free = (
                "unknown"
                if self._mem_free is None
                else f"{self._mem_free / (1 << 20):.0f} MiB"
            )
            size_mib = tensor.numel() * tensor.element_size() / (1 << 20)
            logger.warning(
                f"{rank_prefix()} skipped {what} ({size_mib:.0f} MiB, "
                f"device={tensor.device}): only {free} free on gpu, diagnostics "
                "must not compete with calibration for VRAM"
            )

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------

    def capture_weight_baseline(self, sample_size: int = WEIGHT_SAMPLE_SIZE):
        items = sorted(self.module_names.items(), key=lambda item: item[1])
        if not items:
            return
        stride = max(1, len(items) // sample_size)
        chosen = list(items[::stride][:sample_size])
        chosen.extend(items[: min(8, len(items))])

        seen = set()
        offenders: List[Tuple[str, TensorScan]] = []
        extremes: List[Tuple[float, str]] = []
        for module, name in chosen:
            if id(module) in seen:
                continue
            seen.add(id(module))
            scan = self._weight_scan(module)
            if scan is None:
                continue
            self.baseline[name] = scan
            self.baseline_checked[name] = False
            if not scan.finite:
                offenders.append((name, scan))
            elif scan.absmax == scan.absmax:  # not NaN
                extremes.append((scan.absmax, name))

        prefix = rank_prefix()
        summary = f"{prefix} weight baseline: {len(self.baseline)} modules sampled"
        if extremes:
            summary += (
                f"; absmax range [{min(extremes)[0]:.3g}, {max(extremes)[0]:.3g}]"
            )
        logger.info(summary)
        for name, scan in offenders:
            logger.error(
                f"{prefix} weight baseline: '{name}' is NON-FINITE at calibration "
                f"start (nan={scan.nan}, inf={scan.inf}, dtype={scan.dtype}). The "
                "in-memory model does not match the checkpoint on disk."
            )
        for absmax, name in sorted(extremes, reverse=True)[:3]:
            if absmax > EXTREME_WEIGHT_THRESHOLD:
                logger.warning(
                    f"{prefix} weight baseline: '{name}' has extreme absmax="
                    f"{absmax:.3e} (dtype={self.baseline[name].dtype})"
                )

    @staticmethod
    def _stored_weight(module: nn.Module) -> Optional[torch.Tensor]:
        """Parameter value as stored, without onloading it to a device."""
        with _disable_onloading():
            weight = getattr(module, "weight", None)
        return weight if isinstance(weight, torch.Tensor) else None

    def _weight_scan(self, module: nn.Module) -> Optional[TensorScan]:
        name = self.module_names.get(module, type(module).__name__)
        try:
            weight = self._stored_weight(module)
            if weight is None:
                return None
            if weight.device.type == "meta":
                # disk offload keeps meta tensors resident: read the value to
                # cpu, never to gpu
                with torch.no_grad(), _materialize_on_cpu(module):
                    return scan_tensor(module.weight)
            if self._should_skip_scan(f"weight scan of '{name}'", weight):
                return None
            with torch.no_grad():
                return scan_tensor(weight)
        except Exception as exc:
            logger.debug(
                f"{rank_prefix()} weight scan failed for '{name}': {exc!r}"
            )
            return None

    def _check_baseline_at_first_use(self, module: nn.Module, name: str):
        baseline = self.baseline.get(name)
        if baseline is None or self.baseline_checked.get(name):
            return
        self.baseline_checked[name] = True
        scan = self._weight_scan(module)
        if scan is None:
            return
        changed = scan.finite != baseline.finite
        if not changed and scan.finite and baseline.finite:
            scale = max(1.0, abs(baseline.absmax))
            changed = abs(scan.absmax - baseline.absmax) > 1e-6 * scale
        if changed:
            self.baseline_changed.append(name)
            logger.error(
                f"{rank_prefix()} weight CHANGED between load and first "
                f"calibration use at '{name}': finite {baseline.finite} -> "
                f"{scan.finite}, absmax {baseline.absmax:.6g} -> {scan.absmax:.6g}. "
                "The in-memory weights were modified/corrupted during the run "
                "(check concurrent writers of the offload folder)."
            )

    # ------------------------------------------------------------------
    # activations
    # ------------------------------------------------------------------

    def observe_input(self, module: nn.Module, inp: torch.Tensor):
        if self.disabled:
            return
        try:
            self._observe_input(module, inp)
        except Exception as exc:
            self.disable(f"input tracking failed: {exc!r}")

    def _observe_input(self, module: nn.Module, inp: torch.Tensor):
        if not isinstance(inp, torch.Tensor) or not inp.is_floating_point():
            return

        self.sequence += 1
        record = self.records.get(module)
        if record is None:
            name = self.module_names.get(module, type(module).__name__)
            record = InputRecord(name=name)
            self.records[module] = record
            self._check_baseline_at_first_use(module, name)

        record.calls += 1
        record.dtype = str(inp.dtype)
        record.device = str(inp.device)
        record.shape = tuple(inp.shape)

        if self._should_skip_scan(f"input scan of '{record.name}'", inp):
            return

        scan = scan_tensor(inp)
        if scan.finite:
            if scan.absmax == scan.absmax and not (record.absmax >= scan.absmax):
                record.absmax = scan.absmax
            if scan.absmax > EXTREME_INPUT_THRESHOLD and not record.extreme_reported:
                record.extreme_reported = True
                logger.warning(
                    f"{rank_prefix()} extreme input magnitude at '{record.name}' "
                    f"(call #{record.calls}): absmax={scan.absmax:.3e}, "
                    f"dtype={scan.dtype}, shape={record.shape}. Values this large "
                    "overflow the float32 Hessian when squared."
                )
            return

        record.nonfinite_calls += 1
        if record.first_nonfinite_call >= 0:
            return

        record.first_nonfinite_call = record.calls
        record.first_nonfinite_seq = self.sequence
        record.nan = scan.nan
        record.inf = scan.inf
        record.sample = scan.sample
        logger.error(
            f"{rank_prefix()} first non-finite INPUT at '{record.name}' "
            f"(call #{record.calls}, sequence {self.sequence}): {scan.nan} NaN / "
            f"{scan.inf} Inf of {scan.numel} elements, dtype={scan.dtype}, "
            f"device={scan.device}, shape={record.shape}, absmax of finite part="
            f"{scan.absmax:.3e}, sample={scan.sample}"
        )

    # ------------------------------------------------------------------
    # block and batch probes
    # ------------------------------------------------------------------

    def install_probes(self):
        if self._register_hook is None:
            return
        blocks = self._find_blocks()
        for name, block in blocks:
            self._register_hook(
                block,
                self._make_block_hook(name, "input"),
                hook_type="forward_pre",
            )
            self._register_hook(
                block,
                self._make_block_hook(name, "output"),
                hook_type="forward",
            )
        embed = self._find_input_embedding()
        if embed is not None:
            self._register_hook(embed, self._token_hook, hook_type="forward_pre")
        if blocks:
            self._install_attention_probes(*blocks[0])
            logger.info(
                f"{rank_prefix()} probes installed on {len(blocks)} decoder blocks "
                f"('{blocks[0][0]}' ... '{blocks[-1][0]}')"
            )

    def _install_attention_probes(self, block_name: str, block: nn.Module):
        attention = getattr(block, "self_attn", None)
        direct_paths = (
            "q_a_proj",
            "q_a_layernorm",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
            "kv_b_proj",
        )
        if not isinstance(attention, nn.Module) or not all(
            isinstance(getattr(attention, path, None), nn.Module)
            for path in direct_paths
        ):
            return

        attention_name = f"{block_name}.self_attn"
        self._register_hook(
            attention,
            self._make_attention_hook(f"{attention_name} [input]", "input"),
            hook_type="forward_pre",
        )
        for path in direct_paths:
            site = f"{attention_name}.{path} [output]"
            self._register_hook(
                getattr(attention, path),
                self._make_attention_hook(site, "output"),
                hook_type="forward",
            )

        indexer = getattr(attention, "indexer", None)
        if isinstance(indexer, nn.Module):
            for path in ("wq_b", "wk", "k_norm", "weights_proj"):
                module = getattr(indexer, path, None)
                if isinstance(module, nn.Module):
                    self._register_hook(
                        module,
                        self._make_attention_hook(
                            f"{attention_name}.indexer.{path} [output]", "output"
                        ),
                        hook_type="forward",
                    )

        self._register_hook(
            attention.o_proj,
            self._make_attention_hook(f"{attention_name}.o_proj [input]", "input"),
            hook_type="forward_pre",
        )
        self._register_hook(
            attention,
            self._make_attention_hook(
                f"{attention_name} [output]", "attention_output"
            ),
            hook_type="forward",
        )
        logger.info(
            f"{rank_prefix()} one-shot MLA probes installed on '{attention_name}'"
        )

    def _make_attention_hook(self, site: str, kind: str) -> Callable:
        def pre_hook(module, args):
            self._attention_probe(site, args[0] if args else None)

        def post_hook(module, args, output):
            tensor = (
                output[0] if isinstance(output, (tuple, list)) and output else output
            )
            self._attention_probe(site, tensor)
            if kind == "attention_output":
                self._report_attention_verdict()

        return pre_hook if kind == "input" else post_hook

    def _attention_probe(self, site: str, tensor):
        if (
            self.disabled
            or site in self.attention_records
            or not isinstance(tensor, torch.Tensor)
            or not tensor.is_floating_point()
        ):
            return
        try:
            self.attention_sequence += 1
            scan = None
            if not self._should_skip_scan(f"attention probe at '{site}'", tensor):
                scan = scan_tensor(tensor)
            self.attention_records[site] = AttentionRecord(
                sequence=self.attention_sequence,
                shape=tuple(tensor.shape),
                scan=scan,
            )
        except Exception as exc:
            self.disable(f"attention probe failed: {exc!r}")

    def _report_attention_verdict(self):
        if self.disabled or self.attention_verdict is not None:
            return
        try:
            ordered = sorted(
                self.attention_records.items(), key=lambda item: item[1].sequence
            )
            if not ordered:
                return

            prefix = rank_prefix()
            lines = [f"{prefix} layer-0 MLA first-forward boundaries:"]
            for site, record in ordered:
                scan = record.scan
                if scan is None:
                    stats = "scan skipped for low VRAM"
                else:
                    stats = (
                        f"finite={scan.finite} nan={scan.nan} inf={scan.inf} "
                        f"range=[{scan.minimum:.3e}, {scan.maximum:.3e}] "
                        f"absmax={scan.absmax:.3e} dtype={scan.dtype}"
                    )
                lines.append(
                    f"  {record.sequence:02d} {site}: "
                    f"shape={record.shape} {stats}"
                )
            logger.info("\n".join(lines))

            bad = next(
                (
                    (site, record)
                    for site, record in ordered
                    if record.scan is not None and not record.scan.finite
                ),
                None,
            )
            skipped = any(record.scan is None for _, record in ordered)
            input_site = next(
                (site for site, _ in ordered if site.endswith("self_attn [input]")),
                "",
            )
            o_proj_input_site = next(
                (site for site, _ in ordered if site.endswith("o_proj [input]")),
                "",
            )
            attention_output_site = next(
                (site for site, _ in ordered if site.endswith("self_attn [output]")),
                "",
            )

            if bad is None:
                verdict = (
                    "all observed layer-0 MLA boundaries were finite"
                    if not skipped
                    else (
                        "no observed boundary was non-finite, but one or more "
                        "scans were skipped"
                    )
                )
            elif bad[0] == input_site:
                verdict = "layer-0 attention input was already non-finite"
            elif bad[0] == o_proj_input_site:
                skipped_before_bad = any(
                    record.scan is None and record.sequence < bad[1].sequence
                    for _, record in ordered
                )
                if skipped_before_bad:
                    verdict = (
                        "first detected non-finite boundary is o_proj input, but an "
                        "earlier projection scan was skipped for low VRAM"
                    )
                else:
                    verdict = (
                        "all observed projection boundaries before o_proj were "
                        "finite, but o_proj input was non-finite; corruption began "
                        "inside the unhookable attention math interval "
                        "(reshape/split, RoPE, index kernel/mask, score einsum, "
                        "softmax, or value einsum)"
                    )
            elif bad[0] == attention_output_site:
                o_proj_input = self.attention_records.get(o_proj_input_site)
                if o_proj_input is not None and o_proj_input.scan is not None:
                    verdict = (
                        "o_proj input was finite but layer-0 attention output was "
                        "non-finite; the first observed failure is o_proj"
                    )
                else:
                    verdict = (
                        "layer-0 attention output was non-finite, but the o_proj "
                        "input scan was skipped for low VRAM"
                    )
            else:
                previous = next(
                    (
                        site
                        for site, record in reversed(ordered[: ordered.index(bad)])
                        if record.scan is not None and record.scan.finite
                    ),
                    None,
                )
                verdict = f"first observed non-finite boundary is '{bad[0]}'"
                if previous is not None:
                    verdict += f" after finite boundary '{previous}'"

            self.attention_verdict = verdict
            log = logger.info if bad is None else logger.error
            log(f"{prefix} layer-0 MLA verdict: {verdict}")
        except Exception as exc:
            self.disable(f"attention verdict failed: {exc!r}")

    def _make_block_hook(self, site: str, kind: str) -> Callable:
        def pre_hook(module, args):
            self._probe_scan(f"{site} [{kind}]", args[0] if args else None)

        def post_hook(module, args, output):
            if isinstance(output, (tuple, list)):
                for index, item in enumerate(output[:3]):
                    self._probe_scan(f"{site} [{kind}#{index}]", item)
            else:
                self._probe_scan(f"{site} [{kind}]", output)

        return pre_hook if kind == "input" else post_hook

    def _probe_scan(self, site: str, tensor):
        if self.disabled or not isinstance(tensor, torch.Tensor):
            return
        try:
            self.site_calls[site] = self.site_calls.get(site, 0) + 1
            if site in self.site_reports:
                return
            if self._should_skip_scan(f"block probe at '{site}'", tensor):
                return
            scan = scan_tensor(tensor)
            if scan.finite:
                return
            message = (
                f"first non-finite hidden state at '{site}' (call "
                f"#{self.site_calls[site]}): {scan.nan} NaN / {scan.inf} Inf of "
                f"{scan.numel} elements, dtype={scan.dtype}, "
                f"absmax of finite part={scan.absmax:.3e}"
            )
            self.site_reports[site] = message
            logger.error(f"{rank_prefix()} block probe: {message}")
        except Exception as exc:
            self.disable(f"block probe failed: {exc!r}")

    def _token_hook(self, module, args):
        if self.disabled or not args:
            return
        try:
            tokens = args[0]
            if not isinstance(tokens, torch.Tensor) or tokens.is_floating_point():
                return
            self.batch_index += 1
            minimum = int(tokens.min())
            maximum = int(tokens.max())
            vocab = getattr(module, "num_embeddings", None) or getattr(
                module, "vocab_size", None
            )
            out_of_range = vocab is not None and (maximum >= int(vocab) or minimum < 0)
            if out_of_range:
                logger.error(
                    f"{rank_prefix()} calibration batch #{self.batch_index}: token "
                    f"ids outside embedding range [0, {int(vocab)}) - min={minimum}, "
                    f"max={maximum}, shape={tuple(tokens.shape)}"
                )
            elif self.batch_index <= 10 or self.batch_index % 25 == 0:
                logger.info(
                    f"{rank_prefix()} calibration batch #{self.batch_index}: "
                    f"input_ids shape={tuple(tokens.shape)} min={minimum} "
                    f"max={maximum}"
                )
        except Exception as exc:
            self.disable(f"token probe failed: {exc!r}")

    def _find_blocks(self) -> List[Tuple[str, nn.Module]]:
        groups: Dict[str, Dict[int, Tuple[str, nn.Module]]] = {}
        for name, module in self.model.named_modules():
            match = _LAYER_RE.match(name)
            if match is None:
                continue
            groups.setdefault(match.group("prefix"), {})[int(match.group("index"))] = (
                name,
                module,
            )
        if not groups:
            return []
        prefix = max(groups, key=lambda key: len(groups[key]))
        return [groups[prefix][index] for index in sorted(groups[prefix])]

    def _find_input_embedding(self) -> Optional[nn.Module]:
        try:
            module = self.model.get_input_embeddings()
            if isinstance(module, nn.Module):
                return module
        except Exception:
            pass
        for name, module in self.model.named_modules():
            if name.endswith("embed_tokens") or module.__class__.__name__.endswith(
                "Embedding"
            ):
                return module
        return None

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def log_skip_postmortem(
        self,
        module: nn.Module,
        name: str,
        hessian: torch.Tensor,
        num_samples: int,
    ):
        if self.disabled:
            return
        try:
            self._log_skip_postmortem(module, name, hessian, num_samples)
        except Exception as exc:
            self.disable(f"skip postmortem failed: {exc!r}")

    def _log_skip_postmortem(
        self, module: nn.Module, name: str, hessian: torch.Tensor, num_samples: int
    ):
        prefix = rank_prefix()
        hessian_scan = scan_tensor(hessian)
        record = self.records.get(module)
        weight_scan = None
        if self.skip_weight_audits < SKIP_WEIGHT_AUDIT_LIMIT:
            self.skip_weight_audits += 1
            weight_scan = self._weight_scan(module)

        lines = [
            f"{prefix} DIAGNOSIS for skipped module '{name}' ({num_samples} tokens):",
            f"  hessian: shape={tuple(hessian.shape)} dtype={hessian.dtype} "
            f"device={hessian.device} nan={hessian_scan.nan} inf={hessian_scan.inf} "
            f"finite_absmax={hessian_scan.absmax:.3e}",
        ]
        if record is None:
            lines.append("  inputs: module never invoked during calibration")
        else:
            lines.append(
                f"  inputs: calls={record.calls} dtype={record.dtype} "
                f"device={record.device} shape={record.shape} "
                f"absmax={record.absmax:.3e} nonfinite_calls={record.nonfinite_calls}"
            )
            if record.first_nonfinite_call >= 0:
                lines.append(
                    f"  first non-finite input: call #{record.first_nonfinite_call} "
                    f"(sequence {record.first_nonfinite_seq}), {record.nan} NaN / "
                    f"{record.inf} Inf, sample={record.sample}"
                )
        if weight_scan is None:
            lines.append(
                "  weight: not audited (limit reached, set "
                f"SKIP_WEIGHT_AUDIT_LIMIT>{self.skip_weight_audits} to raise)"
            )
        else:
            lines.append(
                f"  weight at compression time: dtype={weight_scan.dtype} "
                f"finite={weight_scan.finite} nan={weight_scan.nan} "
                f"inf={weight_scan.inf} absmax={weight_scan.absmax:.3e}"
            )

        if record is None:
            verdict = (
                "the module was never invoked by the calibration data: no tokens "
                "reached it (e.g. an expert that was never routed)"
            )
        elif record.nonfinite_calls:
            verdict = (
                "non-finite INPUT activations reached this module: the corruption "
                "is upstream (see the earliest 'first non-finite INPUT' sequence "
                "and block probes)"
            )
        elif record.extreme_reported:
            verdict = (
                f"inputs were huge but finite (absmax={record.absmax:.3e}): the "
                "float32 Hessian overflows on the squared values"
            )
        elif weight_scan is not None and not weight_scan.finite:
            verdict = (
                "the module's own weight is non-finite at compression time: the "
                "in-memory weights were corrupted during the run"
            )
        else:
            verdict = (
                "inputs and weight look finite: the Hessian went non-finite while "
                "accumulating (look for a single extreme batch or partial offload "
                "reads)"
            )
        lines.append(f"  verdict: {verdict}")

        self.skips[name] = verdict
        logger.error("\n".join(lines))

    def log_summary(self):
        if self.disabled:
            return
        try:
            self._log_summary()
        except Exception as exc:
            self.disable(f"summary failed: {exc!r}")

    def _log_summary(self):
        prefix = rank_prefix()
        records = [record for record in self.records.values() if record.calls > 0]
        nonfinite = sorted(
            (record for record in records if record.first_nonfinite_call >= 0),
            key=lambda record: record.first_nonfinite_seq,
        )
        summary = (
            f"{prefix} calibration summary: {len(self.module_names)} quantized "
            f"modules, {len(records)} invoked, {len(nonfinite)} saw non-finite "
            f"inputs, {len(self.skips)} skipped, {len(self.baseline_changed)} "
            "sampled weights changed in memory"
        )
        if self.mem_skips:
            summary += f", {self.mem_skips} scans skipped for low VRAM"
        logger.info(summary)
        for record in nonfinite[:20]:
            logger.error(
                f"{prefix} non-finite input '{record.name}' first at call "
                f"#{record.first_nonfinite_call} (sequence "
                f"{record.first_nonfinite_seq}, shape={record.shape}): "
                f"{record.nan} NaN / {record.inf} Inf, sample={record.sample}"
            )
        if len(nonfinite) > 20:
            logger.error(
                f"{prefix} ... and {len(nonfinite) - 20} more modules saw "
                "non-finite inputs (earliest ones listed first)"
            )
        for site, message in sorted(self.site_reports.items()):
            logger.error(f"{prefix} block probe: {message}")

        invoked = sorted((record.calls, record.name) for record in records)
        if invoked:
            logger.info(
                f"{prefix} module call counts: min={invoked[0][0]} "
                f"max={invoked[-1][0]}; lowest: "
                f"{', '.join(f'{name}({calls})' for calls, name in invoked[:5])}"
            )
        unseen = [
            name
            for module, name in self.module_names.items()
            if module not in self.records
        ]
        if unseen:
            preview = ", ".join(unseen[:10])
            logger.warning(
                f"{prefix} {len(unseen)} quantized modules were never invoked "
                f"during calibration: {preview}"
                + (" ..." if len(unseen) > 10 else "")
            )

    def format_skip_table(self) -> str:
        lines = [
            f"{rank_prefix()} {len(self.skips)} skipped modules (diagnosis per module):"
        ]
        for name in sorted(self.skips):
            lines.append(f"  {name}: {self.skips[name]}")
        return "\n".join(lines)
