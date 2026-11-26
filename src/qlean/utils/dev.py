# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Ported from: llmcompressor/utils/dev.py (skip_weights_initialize,
#   skip_weights_download, patch_transformers_logger_level, get_main_device,
#   get_high_precision, dispatch_for_generation, load_context)
# Ported from: llmcompressor/utils/compressed_tensors_utils.py
#   (suspend_distributed_timeout)
# Ported from: llmcompressor/datasets/utils.py (get_rank_partition)
# temporary_offload_folder is a qlean implementation with no upstream counterpart.

from __future__ import annotations

import contextlib
import datetime
import logging
import math
import os
import shutil
import tempfile
import warnings
from typing import TYPE_CHECKING, Iterator, Type

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

__all__ = [
    "skip_weights_initialize",
    "skip_weights_download",
    "patch_transformers_logger_level",
    "get_main_device",
    "get_high_precision",
    "dispatch_for_generation",
    "load_context",
    "temporary_offload_folder",
    "suspend_distributed_timeout",
    "get_rank_partition",
]


def _default_model_cls():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def _patch_attr():
    from compressed_tensors.utils import patch_attr

    return patch_attr


def _torch_init_functions():
    try:
        # Transformers < v5 support
        from transformers.modeling_utils import TORCH_INIT_FUNCTIONS
    except ImportError:
        # Transformers v5 support
        from transformers.initialization import TORCH_INIT_FUNCTIONS

    return TORCH_INIT_FUNCTIONS


@contextlib.contextmanager
def load_context(model_cls: Type[PreTrainedModel] | None = None):
    """Context manager for loading HuggingFace models with offloading and MoE support.

    Combines ``compressed_tensors.offload.load_offloaded_model`` with
    ``qlean.modeling.moe.linearize.load_quantizable_moe`` so models that
    require disk offloading and/or MoE linearization can be loaded with a single
    context.

    :param model_cls: model class to patch, defaults to ``AutoModelForCausalLM``
    """
    from qlean.utils.optional import require_gptq

    require_gptq()

    from compressed_tensors.offload import load_offloaded_model

    from qlean.modeling.moe.linearize import load_quantizable_moe

    if model_cls is None:
        model_cls = _default_model_cls()

    with contextlib.ExitStack() as stack:
        stack.enter_context(load_offloaded_model(model_cls))
        stack.enter_context(load_quantizable_moe(model_cls))
        yield


@contextlib.contextmanager
def temporary_offload_folder(parent_dir: str | None = None) -> Iterator[str]:
    """Create a rank-coordinated temporary offload directory.

    In multi-node runs, ``parent_dir`` must be a writable directory shared by all
    ranks.
    """
    is_distributed = dist.is_initialized()
    is_main_process = not is_distributed or dist.get_rank() == 0
    folder: list[str | None] = [None]

    if is_main_process:
        folder[0] = tempfile.mkdtemp(prefix="qlean_offload_", dir=parent_dir)
    try:
        if is_distributed:
            dist.broadcast_object_list(folder, src=0)

        offload_folder = folder[0]
        if offload_folder is None:
            raise RuntimeError("Distributed offload directory was not initialized")
    except BaseException:
        # Setup failed before the coordinated cleanup below could run. Remove the
        # directory locally: a barrier here would hang on an already-broken group.
        if is_main_process and folder[0] is not None:
            shutil.rmtree(folder[0], ignore_errors=True)
        raise

    try:
        yield offload_folder
    finally:
        try:
            if is_distributed:
                dist.barrier()
        finally:
            if is_main_process:
                shutil.rmtree(offload_folder, ignore_errors=True)
        if is_distributed:
            # The group may already be broken when the body failed; a second
            # barrier would then hang instead of letting the cleanup finish.
            with contextlib.suppress(Exception):
                dist.barrier()


@contextlib.contextmanager
def skip_weights_download(model_class: Type[PreTrainedModel] | None = None):
    """Context manager under which models are initialized without downloading weights.

    Differs from ``init_empty_weights`` in that weights are allocated on their
    assigned devices with random values, rather than on the meta device.

    :param model_class: class to patch, defaults to ``AutoModelForCausalLM``
    """
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

    if model_class is None:
        model_class = _default_model_cls()

    original_fn = model_class.from_pretrained
    weights_files = [
        "*.bin",
        "*.safetensors",
        "*.pth",
        SAFE_WEIGHTS_INDEX_NAME,
        WEIGHTS_INDEX_NAME,
        "*.msgpack",
        "*.pt",
    ]

    @classmethod
    def patched(cls, *args, **kwargs):
        nonlocal tmp_dir

        from transformers.conversion_mapping import extract_weight_conversions_for_model

        # intercept model stub
        model_stub = args[0] if args else kwargs.pop("pretrained_model_name_or_path")

        # download non-weight files into tmp dir
        os.makedirs(tmp_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_stub, local_dir=tmp_dir, ignore_patterns=weights_files
        )

        # make an empty weights file to avoid errors
        weights_file_path = os.path.join(tmp_dir, "model.safetensors")
        save_file({}, weights_file_path, metadata={"format": "pt"})

        # load from tmp dir
        model = original_fn(tmp_dir, *args[1:], **kwargs)

        # replace model path
        model.name_or_path = model_stub
        model.config._name_or_path = model_stub

        # normally transformers populates ``_weight_conversions`` based on which
        # were used since none were used (no weights loaded), populate directly
        # so saving in original checkpoint format still works
        model._weight_conversions = extract_weight_conversions_for_model(model)

        return model

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        _patch_attr()(model_class, "from_pretrained", patched),
        skip_weights_initialize(),
        patch_transformers_logger_level(),
    ):
        yield


@contextlib.contextmanager
def skip_weights_initialize(use_zeros: bool = False):
    """Context manager that skips PyTorch weight initialization.

    Similar to ``transformers.model_utils.no_init_weights``, but also patches
    ``torch.Tensor`` initialization functions so that tensors allocated on real
    devices (not just meta) keep their existing values.

    :param use_zeros: if ``True``, initialize skipped tensors to zeros instead of
        leaving them untouched
    """

    patch_attr = _patch_attr()
    torch_init_functions = _torch_init_functions()

    def skip(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if use_zeros:
            return tensor.fill_(0)
        return tensor

    with contextlib.ExitStack() as stack:
        for name in torch_init_functions.keys():
            stack.enter_context(patch_attr(torch.nn.init, name, skip))
            stack.enter_context(patch_attr(torch.Tensor, name, skip))
        yield


@contextlib.contextmanager
def patch_transformers_logger_level(level: int = logging.ERROR):
    """Context under which the transformers logger level is modified.

    Useful with ``skip_weights_download`` to squelch warnings about missing
    parameters in the checkpoint.

    :param level: new logging level for the transformers modeling logger
    """
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    # Save the logger's own level, not getEffectiveLevel(): restoring an inherited
    # level would pin NOTSET to a concrete value and silence later set_verbosity.
    restore_log_level = transformers_logger.level

    transformers_logger.setLevel(level=level)
    try:
        yield
    finally:
        transformers_logger.setLevel(level=restore_log_level)


def get_main_device() -> torch.device:
    """Return the main device for the current process."""
    is_distributed_enable = torch.distributed.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_distributed_enable else 0

    # Check for unsupported MPS + distributed combination
    if hasattr(torch, "mps") and torch.mps.is_available():
        if is_distributed_enable:
            raise RuntimeError("Parallelism has not been supported for MPS")
        return torch.device("mps")

    elif torch.accelerator.is_available():
        accel_type = torch.accelerator.current_accelerator().type
        return torch.device(accel_type, local_rank)
    else:
        logger.warning("No accelerator available! Compressing model on CPU instead")
        return torch.device("cpu")


def get_high_precision() -> torch.dtype:
    """Return a high-precision dtype suitable for the current device."""
    main_device = get_main_device()

    if main_device.type == "mps":  # MPS does not support float64
        return torch.float32

    return torch.float64


def dispatch_for_generation(*args, **kwargs) -> PreTrainedModel:
    """Dispatch a model for autoregressive generation.

    Modules are dispatched evenly across available devices and kept onloaded if
    possible.

    Deprecated: use ``compressed_tensors.offload.dispatch_model`` directly.
    """
    warnings.warn(
        "dispatch_for_generation is deprecated; use "
        "compressed_tensors.offload::dispatch_model instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from compressed_tensors.offload import dispatch_model

    return dispatch_model(*args, **kwargs)


@contextlib.contextmanager
def suspend_distributed_timeout(
    timeout: datetime.timedelta = datetime.timedelta(hours=3),
    current_group: dist.ProcessGroup | None = None,
):
    """
    Context manager that extends the timeout for distributed operations.

    Creates a temporary process group with an extended timeout to prevent
    timeout errors during long-running operations (e.g., model saving) in
    distributed training environments. The context manager synchronizes all
    processes before and after the operation using barriers.

    :param timeout: The extended timeout for the temporary process group.
        Defaults to 3 hours
    :param current_group: The current process group to synchronize. If None,
        defaults to dist.group.WORLD
    """
    if not dist.is_initialized():
        yield None
        return

    if current_group is None:
        current_group = dist.group.WORLD
    suspend_group = dist.new_group(backend="gloo", timeout=timeout)

    try:
        # CPU-side sync first: ranks may finish compress_model/to_accelerate at
        # different times. A gloo barrier with a long timeout prevents the
        # subsequent NCCL barrier from timing out while waiting for stragglers.
        dist.barrier(group=suspend_group)
        dist.barrier(group=current_group)
        yield suspend_group
    finally:
        try:
            dist.barrier(group=suspend_group)
            dist.barrier(group=current_group)
        finally:
            dist.destroy_process_group(group=suspend_group)


def get_rank_partition(split: str, num_samples: int) -> str:
    """Return the calibration-data slice assigned to this distributed rank."""
    if "[" in split:
        raise ValueError("Split string should not already contain partitioning brackets")

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    start = math.floor(num_samples * rank / world_size)
    end = math.floor(num_samples * (rank + 1) / world_size)
    return f"{split}[{start}:{end}]"
