from __future__ import annotations

import functools
import math
import os
import random
import warnings
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from enum import Enum
from functools import partial
from types import MethodType
from typing import Any, ClassVar, Literal

import einops
import numpy as np
import torch
from torch import nn
from torch.nn.modules.transformer import Module, Tensor
from torch.utils.checkpoint import checkpoint

# TODO(eddiebergman): Make this an option
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
SAVE_PEAK_MEM_FACTOR = 8

# TODO(eddiebergman): pulled from `def _estimate_model_usage()`
CONSTANT_MEMORY_OVERHEAD = 100_000_000
MEMORY_FACTOR_SAVE_PEAK_MEM_ACTIVE = 2.5
DEFAULT_CPU_MEMORY_GB_IF_NOT_CUDA = 8

# TODO(eddiebergman): pulled from `def _estimate_model_usage()`
# Had it's own todo of "check if correct"
NUM_SAMPLES_FACTOR = 4
NUM_SAMPLES_PLUS_FEATURES = 6.5
CELLS_FACTOR = 0.25
CELLS_SQUARED_FACTOR = 1.3e-7

TO_BYTES_CONVERSION = {"b": 1, "mb": 1e6, "gb": 1e9}


def support_save_peak_mem_factor(method: MethodType) -> Callable:
    """Can be applied to a method acting on a tensor 'x' whose first dimension is a
    flat batch dimension
    (i.e. the operation is trivially parallel over the first dimension).

    For additional tensor arguments, it is assumed that the first dimension is again
    the batch dimension, and that non-tensor arguments can be passed as-is
    to splits when parallelizing over the batch dimension.

    The decorator adds options 'add_input' to add the principal input 'x' to the
    result of the method and 'allow_inplace'.
    By setting 'allow_inplace', the caller indicates that 'x'
    is not used after the call and its buffer can be reused for the output.

    Setting 'allow_inplace' does not ensure that the operation will be inplace,
    and the return value should be used for clarity and simplicity.

    Moreover, it adds an optional int parameter 'save_peak_mem_factor' that is
    only supported in combination with 'allow_inplace' during inference and subdivides
    the operation into the specified number of chunks to reduce peak memory consumption.
    """

    def method_(
        self: torch.nn.Module,
        x: torch.Tensor,
        *args: torch.Tensor,
        add_input: bool = False,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert isinstance(self, torch.nn.Module)
        assert save_peak_mem_factor is None or allow_inplace, (
            "The parameter save_peak_mem_factor only supported with 'allow_inplace' set."
        )
        assert isinstance(x, torch.Tensor)

        tensor_inputs = list(tuple(self.parameters()) + tuple(args))

        assert (
            save_peak_mem_factor is None
            or not any(t.requires_grad for t in tensor_inputs)
            or not torch.is_grad_enabled()
        ), "The parameter save_peak_mem_factor is only supported during inference."

        if save_peak_mem_factor is not None:
            assert isinstance(save_peak_mem_factor, int)
            assert save_peak_mem_factor > 1
            split_size = (x.size(0) + save_peak_mem_factor - 1) // save_peak_mem_factor

            split_args = zip(
                *[
                    torch.split(arg, split_size)
                    if isinstance(arg, torch.Tensor)
                    else [arg] * save_peak_mem_factor
                    for arg in (x, *args)
                ],
            )

            for x_, *args_ in split_args:
                if add_input:
                    x_[:] += method(self, x_, *args_, **kwargs)
                else:
                    x_[:] = method(self, x_, *args_, **kwargs)
            return x

        if add_input:
            return x + method(self, x, *args, **kwargs)

        return method(self, x, *args, **kwargs)

    return method_


class MemoryUsageEstimator:
    SAVE_PEAK_MEM_FACTOR = 8

    @classmethod
    def convert_units(
        cls,
        value: float,
        from_unit: Literal["b", "mb", "gb"],
        to_unit: Literal["b", "mb", "gb"],
    ) -> float:
        """Convert a value from one unit to another."""
        if from_unit not in TO_BYTES_CONVERSION:
            raise ValueError(
                f"Invalid unit {from_unit}. Must be one of 'b', 'mb', or 'gb'.",
            )
        if to_unit not in TO_BYTES_CONVERSION:
            raise ValueError(
                f"Invalid unit {to_unit}. Must be one of 'b', 'mb', or 'gb'.",
            )

        return (value * TO_BYTES_CONVERSION[from_unit]) / TO_BYTES_CONVERSION[to_unit]

    @classmethod
    def convert_bytes_to_unit(
        cls,
        value: float,
        unit: Literal["b", "mb", "gb"],
    ) -> float:
        """Convenience method to convert bytes to a different unit.

        Args:
            value: The number of bytes.
            unit: The unit to convert to.

        Returns:
            The number of bytes in the new unit.
        """
        return cls.convert_units(value, "b", unit)

    @classmethod
    def estimate_memory_of_one_batch(
        cls,
        X: torch.Tensor,
        model: torch.nn.Module,
        *,
        cache_kv: bool,
        dtype_byte_size: int,
        unit: Literal["b", "mb", "gb"] = "gb",
        n_train_samples: int | None = None,
    ) -> float:
        """Estimate the memory usage of a single batch.

        The calculation is done based on the assumption that save_peak_mem_factor
        is not used (since this estimation is used to determine whether to use it).

        Args:
            X: The input tensor.
            model: The model to estimate the memory usage of.
            cache_kv: Whether key and value tensors are cached.
            dtype_byte_size: The size of the data type in bytes.
            unit: The unit to convert the memory usage to.
            n_train_samples: The number of training samples (only for cache_kv mode)

        Returns:
            The estimated memory usage of a single batch.
        """
        if cache_kv:
            assert isinstance(
                n_train_samples,
                int,
            ), "n_train_samples must be provided when cache_kv is True"

        if unit not in TO_BYTES_CONVERSION:
            raise ValueError(f"Invalid unit {unit}. Must be one of 'b', 'mb', or 'gb'.")

        embedding_size = model.ninp
        features_per_group = model.features_per_group

        n_layers = None
        # Assumes the model has only encoder blocks
        if (
            hasattr(model, "transformer_encoder")
            and model.transformer_encoder is not None
        ):
            n_layers = len(model.transformer_encoder.layers)

        # Guarding against future changes in the transformer model
        # Ideally, there should be an API exposed in the model to get the
        # number of layers
        if n_layers is None:
            n_layers = 12
            warnings.warn(
                "Could not estimate number of encoder/decoder layers in the "
                "transformer model, defaulting to 12.",
                stacklevel=2,
            )

        n_samples, n_features = X.shape[0], X.shape[-1]
        n_feature_groups = int(np.ceil(n_features / features_per_group)) + 1

        model_mem = sum(p.numel() for p in model.parameters()) * dtype_byte_size
        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = (
            n_samples * n_feature_groups * embedding_size * n_layers * dtype_byte_size
        )

        total_mem_bytes = model_mem + X_mem + activation_mem

        if cache_kv:
            cached_mem = (
                n_train_samples  # type: ignore
                * n_feature_groups
                * embedding_size
                * 2  # key and value
                * n_layers
                * dtype_byte_size
            )
            total_mem_bytes += cached_mem

        return cls.convert_bytes_to_unit(total_mem_bytes, unit)

    @classmethod
    def get_max_free_memory(
        cls,
        device: torch.device,
        *,
        unit: Literal["b", "mb", "gb"] = "gb",
        default_gb_cpu_if_failed_to_calculate: float,
    ) -> float:
        """How much memory to use at most in GB, the memory usage will be calculated
        based on an estimation of the systems free memory.

        For CUDA will use the free memory of the GPU. For CPU will default to 32 GB.

        Returns:
        -------
        The maximum memory usage in GB.
        """
        # TODO(Arjun): Make it accept a value for GPU specified by the user

        # TODO: Get System Stats and adapt to free memory for default case

        if device.type.startswith("cpu"):
            try:
                free_memory = (
                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
                )
            except AttributeError:
                from tabpfn.utils import get_total_memory_windows

                if os.name == "nt":
                    free_memory = get_total_memory_windows()
                else:
                    warnings.warn(
                        "Could not get system memory, defaulting to"
                        f" {default_gb_cpu_if_failed_to_calculate} GB",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    free_memory = cls.convert_units(
                        default_gb_cpu_if_failed_to_calculate,
                        "gb",
                        "b",
                    )

            except ValueError:
                warnings.warn(
                    "Could not get system memory, defaulting to"
                    f" {default_gb_cpu_if_failed_to_calculate} GB",
                    RuntimeWarning,
                    stacklevel=2,
                )
                free_memory = cls.convert_units(
                    default_gb_cpu_if_failed_to_calculate,
                    "gb",
                    "b",
                )

        elif device.type.startswith("cuda"):
            t = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            free_memory = t - a  # free inside reserved
        elif device.type.startswith("mps"):
            # It seems like we would want to use the following functions:
            #    * torch.mps.recommended_max_memory
            #    * torch.mps.current_allocated_memory
            # Not entirely sure of the behavior of the first function
            # so this might not work as intended.
            raise NotImplementedError(
                "Memory estimation for MPS devices is not currently supported."
                " If you have experience with getting memory information for MPS"
                " devices, we would gladly appreciate a contribution!"
                "",
            )
        else:
            raise ValueError(f"Unknown device {device}")

        return cls.convert_bytes_to_unit(free_memory, unit)

    @classmethod
    def estimate_memory_remainder_after_batch(
        cls,
        X: torch.Tensor,
        model: torch.nn.Module,
        *,
        cache_kv: bool,
        device: torch.device,
        dtype_byte_size: int,
        safety_factor: float,
        n_train_samples: int | None = None,
        max_free_mem: float | int | None = None,
    ) -> float:
        """Whether to save peak memory or not.

        Args:
            X: The input tensor.
            model: The model to estimate the memory usage of.
            cache_kv: Whether key and value tensors are cached.
            device: The device to use.
            dtype_byte_size: The size of the data type in bytes.
            safety_factor: The safety factor to apply.
            n_train_samples: The number of training samples (only for cache_kv mode)
            max_free_mem: The amount of free memory available.

        Returns:
            The amount of free memory available after a batch is computed.
        """
        if max_free_mem is None:
            max_free_mem = cls.get_max_free_memory(
                device,
                unit="gb",
                default_gb_cpu_if_failed_to_calculate=DEFAULT_CPU_MEMORY_GB_IF_NOT_CUDA,
            )

        mem_per_batch = cls.estimate_memory_of_one_batch(
            X,
            model,
            cache_kv=cache_kv,
            dtype_byte_size=dtype_byte_size,
            unit="gb",
            n_train_samples=n_train_samples,
        )

        return max_free_mem - (mem_per_batch * safety_factor)

    @classmethod
    def reset_peak_memory_if_required(
        cls,
        save_peak_mem: bool | Literal["auto"] | float | int,
        model: torch.nn.Module,
        X: torch.Tensor,
        *,
        cache_kv: bool,
        device: torch.device,
        dtype_byte_size: int,
        safety_factor: float = 5.0,
        n_train_samples: int | None = None,
    ) -> None:
        """Reset the peak memory if required.

        Args:
            save_peak_mem (bool | "auto" | float | int): If bool, specifies whether to
                save peak memory or not.
                If "auto", the amount of free memory is estimated and the option is
                enabled or disabled based on the estimated usage.
                If float or int, it is considered as the amount of memory available
                (in GB) explicitly specified by the user. In this case, this value is
                used to estimate whether or not to save peak memory.
            model (torch.nn.Module): The model to reset the peak memory of.
            X (torch.Tensor): The input tensor.
            cache_kv (bool): Whether key and value tensors are cached.
            device (torch.device): The device to use.
            dtype_byte_size (int): The size of the data type in bytes.
            safety_factor (float): The safety factor to apply.
            n_train_samples (int): The number of training samples (to be used
                only for cache_kv mode)
        """
        save_peak_mem_is_num = isinstance(
            save_peak_mem,
            (float, int),
        ) and not isinstance(save_peak_mem, bool)
        if save_peak_mem == "auto" or save_peak_mem_is_num:
            memory_available_after_batch = cls.estimate_memory_remainder_after_batch(
                X,
                model,
                cache_kv=cache_kv,
                device=device,
                dtype_byte_size=dtype_byte_size,
                safety_factor=safety_factor,
                n_train_samples=n_train_samples,
                max_free_mem=save_peak_mem
                if isinstance(save_peak_mem, (float, int))
                else None,
            )
            save_peak_mem = memory_available_after_batch < 0

        if save_peak_mem:
            model.reset_save_peak_mem_factor(cls.SAVE_PEAK_MEM_FACTOR)
        else:
            model.reset_save_peak_mem_factor(None)


# usage of custom implementations is required to support ONNX export
def torch_nansum(x: torch.Tensor, axis=None, keepdim=False, dtype=None):
    nan_mask = torch.isnan(x)
    masked_input = torch.where(
        nan_mask,
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
        x,
    )
    return torch.sum(masked_input, axis=axis, keepdim=keepdim, dtype=dtype)


def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: bool = False,
    include_inf: bool = False,
):
    nan_mask = torch.isnan(x)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))

    num = torch.where(nan_mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        axis=axis,
    )
    value = torch.where(nan_mask, torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    if return_nanshare:
        return value / num, 1.0 - num / x.shape[axis]
    return value / num.clip(min=1.0)


def torch_nanstd(x: torch.Tensor, axis: int = 0):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        axis=axis,
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    mean = value / num
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis),
        x.shape[axis],
        dim=axis,
    )
    return torch.sqrt(
        torch_nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1),  # type: ignore
    )


def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: bool = False,
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Normalize data to mean 0 and std 1.

    Args:
        data: The data to normalize. (T, B, H)
        normalize_positions: If > 0, only use the first `normalize_positions` positions for normalization.
        return_scaling: If True, return the scaling parameters as well (mean, std).
        std_only: If True, only divide by std.
        clip: If True, clip the data to [-100, 100].
        mean: If given, use this value instead of computing it.
        std: If given, use this value instead of computing it.
    """
    # TODO(eddiebergman): I feel like this function is easier to just do what you need
    # where you need it, rather than supporting all these variations
    assert (mean is None) == (std is None), (
        "Either both or none of mean and std must be given"
    )
    if mean is None:
        if normalize_positions is not None and normalize_positions > 0:
            mean = torch_nanmean(data[:normalize_positions], axis=0)  # type: ignore
            std = torch_nanstd(data[:normalize_positions], axis=0) + 1e-20
        else:
            mean = torch_nanmean(data, axis=0)  # type: ignore
            std = torch_nanstd(data, axis=0) + 1e-20

        if len(data) == 1 or normalize_positions == 1:
            std[:] = 1.0

        if std_only:
            mean[:] = 0  # type: ignore
    data = (data - mean) / std

    if clip:
        data = torch.clip(data, min=-100, max=100)

    if return_scaling:
        return data, (mean, std)  # type: ignore
    return data


def select_features(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """Select features from the input tensor based on the selection mask,
    and arrange them contiguously in the last dimension.
    If batch size is bigger than 1, we pad the features with zeros to make the number of features fixed.

    Args:
        x: The input tensor of shape (sequence_length, batch_size, total_features)
        sel: The boolean selection mask indicating which features to keep of shape (batch_size, total_features)

    Returns:
        The tensor with selected features.
        The shape is (sequence_length, batch_size, number_of_selected_features) if batch_size is 1.
        The shape is (sequence_length, batch_size, total_features) if batch_size is greater than 1.
    """
    B, total_features = sel.shape
    sequence_length = x.shape[0]

    # If B == 1, we don't need to append zeros, as the number of features don't need to be fixed.
    if B == 1:
        return x[:, :, sel[0]]

    new_x = torch.zeros(
        (sequence_length, B, total_features),
        device=x.device,
        dtype=x.dtype,
    )

    # For each batch, compute the number of selected features.
    sel_counts = sel.sum(dim=-1)  # shape: (B,)

    for b in range(B):
        s = int(sel_counts[b])
        if s > 0:
            new_x[:, b, :s] = x[:, b, sel[b]]

    return new_x


def remove_outliers(
    X: torch.Tensor,
    n_sigma: float = 4,
    normalize_positions: int = -1,
    lower: None | torch.Tensor = None,
    upper: None | torch.Tensor = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    # Expects T, B, H
    assert (lower is None) == (upper is None), "Either both or none of lower and upper"
    assert len(X.shape) == 3, "X must be T,B,H"
    # for b in range(X.shape[1]):
    # for col in range(X.shape[2]):
    if lower is None:
        data = X if normalize_positions == -1 else X[:normalize_positions]
        data_clean = data[:].clone()
        data_mean, data_std = torch_nanmean(data, axis=0), torch_nanstd(data, axis=0)
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

        data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan
        data_mean, data_std = (
            torch_nanmean(data_clean, axis=0),
            torch_nanstd(data_clean, axis=0),
        )
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    return X, (lower, upper)


class InputEncoder(nn.Module):
    """Base class for input encoders.

    All input encoders should subclass this class and implement the `forward` method.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, single_eval_pos: int) -> torch.Tensor:
        """Encode the input tensor.

        Args:
            x: The input tensor to encode.
            single_eval_pos: The position to use for single evaluation.

        Returns:
            The encoded tensor.
        """
        raise NotImplementedError


class SequentialEncoder(nn.Sequential, InputEncoder):
    """An encoder that applies a sequence of encoder steps.

    SequentialEncoder allows building an encoder from a sequence of EncoderSteps.
    The input is passed through each step in the provided order.
    """

    def __init__(self, *args: SeqEncStep, output_key: str = "output", **kwargs: Any):
        """Initialize the SequentialEncoder.

        Args:
            *args: A list of SeqEncStep instances to apply in order.
            output_key:
                The key to use for the output of the encoder in the state dict.
                Defaults to "output", i.e. `state["output"]` will be returned.
            **kwargs: Additional keyword arguments passed to `nn.Sequential`.
        """
        super().__init__(*args, **kwargs)
        self.output_key = output_key

    def forward(self, input: dict, **kwargs: Any) -> torch.Tensor:
        """Apply the sequence of encoder steps to the input.

        Args:
            input:
                The input state dictionary.
                If the input is not a dict and the first layer expects one input key,
                the input tensor is mapped to the key expected by the first layer.
            **kwargs: Additional keyword arguments passed to each encoder step.

        Returns:
            The output of the final encoder step.
        """
        # If the input is not a dict and the first layer expects one input, mapping the
        #   input to the first input key must be correct
        if not isinstance(input, dict) and len(self[0].in_keys) == 1:
            input = {self[0].in_keys[0]: input}

        for module in self:
            input = module(input, **kwargs)

        return input[self.output_key] if self.output_key is not None else input


class SeqEncStep(nn.Module):
    """Abstract base class for sequential encoder steps.

    SeqEncStep is a wrapper around a module that defines the expected input keys
    and the produced output keys. The outputs are assigned to the output keys
    in the order specified by `out_keys`.

    Subclasses should either implement `_forward` or `_fit` and `_transform`.
    Subclasses that transform `x` should always use `_fit` and `_transform`,
    creating any state that depends on the train set in `_fit` and using it in `_transform`.
    This allows fitting on data first and doing inference later without refitting.
    Subclasses that work with `y` can alternatively use `_forward` instead.
    """

    def __init__(
        self,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Initialize the SeqEncStep.

        Args:
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys

    # Either implement _forward:

    def _forward(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Forward pass of the encoder step.

        Implement this if not implementing _fit and _transform.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The output tensor or a tuple of output tensors.
        """
        raise NotImplementedError()

    # Or implement _fit and _transform:

    def _fit(
        self,
        *x: torch.Tensor,
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit the encoder step on the training set.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.
        """
        raise NotImplementedError

    def _transform(
        self,
        *x: torch.Tensor,
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        """Transform the data using the fitted encoder step.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The transformed output tensor or a tuple of output tensors.
        """
        raise NotImplementedError

    def forward(
        self,
        state: dict,
        cache_trainset_representation: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Perform the forward pass of the encoder step.

        Args:
            state: The input state dictionary containing the input tensors.
            cache_trainset_representation:
                Whether to cache the training set representation. Only supported for
                _fit and _transform (not _forward).
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The updated state dictionary with the output tensors assigned to the output keys.
        """
        args = [state[in_key] for in_key in self.in_keys]
        if hasattr(self, "_fit"):
            if kwargs["single_eval_pos"] or not cache_trainset_representation:
                self._fit(*args, **kwargs)
            out = self._transform(*args, **kwargs)
        else:
            assert not cache_trainset_representation
            out = self._forward(*args, **kwargs)

        assert isinstance(out, tuple)
        assert len(out) == len(self.out_keys)
        state.update({out_key: out[i] for i, out_key in enumerate(self.out_keys)})
        return state


class LinearInputEncoderStep(SeqEncStep):
    """A simple linear input encoder step."""

    def __init__(
        self,
        *,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("output",),
    ):
        """Initialize the LinearInputEncoderStep.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            replace_nan_by_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
            in_keys: The keys of the input tensors. Defaults to ("main",).
            out_keys: The keys to assign the output tensors to. Defaults to ("output",).
        """
        super().__init__(in_keys, out_keys)
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    def _fit(self, *x: torch.Tensor, **kwargs: Any):
        """Fit the encoder step. Does nothing for LinearInputEncoderStep."""

    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Apply the linear transformation to the input.

        Args:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return (self.layer(x),)


class NanHandlingEncoderStep(SeqEncStep):
    """Encoder step to handle NaN and infinite values in the input."""

    nan_indicator = -2.0
    inf_indicator = 2.0
    neg_inf_indicator = 4.0

    def __init__(
        self,
        keep_nans: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main", "nan_indicators"),
    ):
        """Initialize the NanHandlingEncoderStep.

        Args:
            keep_nans: Whether to keep NaN values as separate indicators. Defaults to True.
            in_keys: The keys of the input tensors. Must be a single key.
            out_keys: The keys to assign the output tensors to.
        """
        assert len(in_keys) == 1, "NanHandlingEncoderStep expects a single input key"
        super().__init__(in_keys, out_keys)
        self.keep_nans = keep_nans
        self.register_buffer("feature_means_", torch.tensor([]), persistent=False)

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
        """Compute the feature means on the training set for replacing NaNs.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).
        """
        self.feature_means_ = torch_nanmean(x[:single_eval_pos], axis=0)

    def _transform(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace NaN and infinite values in the input tensor.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor and optionally the NaN indicators.
        """
        nans_indicator = None
        if self.keep_nans:
            # TODO: There is a bug here: The values arriving here are already mapped to nan if they were inf before
            nans_indicator = (
                torch.isnan(x) * NanHandlingEncoderStep.nan_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == 1)
                * NanHandlingEncoderStep.inf_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
                * NanHandlingEncoderStep.neg_inf_indicator
            ).to(x.dtype)

        nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
        # replace nans with the mean of the corresponding feature
        x = x.clone()  # clone to avoid inplace operations
        x[nan_mask] = self.feature_means_.unsqueeze(0).expand_as(x)[nan_mask]

        return x, nans_indicator


class RemoveEmptyFeaturesEncoderStep(SeqEncStep):
    """Encoder step to remove empty (constant) features.
    Was changed to NOT DO ANYTHING, the removal of empty features now
    done elsewhere, but the saved model still needs this encoder step.
    TODO: REMOVE.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the RemoveEmptyFeaturesEncoderStep.

        Args:
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.sel = None

    def _fit(self, x: torch.Tensor, **kwargs: Any) -> None:
        """Compute the feature selection mask on the training set.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).
        """
        # self.sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)

    def _transform(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Remove empty features from the input tensor.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor with empty features removed.
        """
        # return (select_features(x, self.sel),)
        return (x,)


class RemoveDuplicateFeaturesEncoderStep(SeqEncStep):
    """Encoder step to remove duplicate features."""

    def __init__(self, normalize_on_train_only: bool = True, **kwargs: Any):
        """Initialize the RemoveDuplicateFeaturesEncoderStep.

        Args:
            normalize_on_train_only: Whether to normalize only on the training set.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
        """Currently does nothing. Fit functionality not implemented."""

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        """Remove duplicate features from the input tensor.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the input tensor (removal not implemented).
        """
        # TODO: This uses a lot of memory, as it computes the covariance matrix for each batch
        #   This could be done more efficiently, models go OOM with this
        return (x,)
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        x_norm = normalize_data(x[:, :normalize_position])
        sel = torch.zeros(x.shape[1], x.shape[2], dtype=torch.bool, device=x.device)
        for B in range(x_norm.shape[1]):
            cov_mat = (torch.cov(x_norm[:, B].transpose(1, 0)) > 0.999).float()
            cov_mat_sum_below_trace = torch.triu(cov_mat).sum(dim=0)
            sel[B] = cov_mat_sum_below_trace == 1.0

        new_x = select_features(x, sel)

        return (new_x,)


class VariableNumFeaturesEncoderStep(SeqEncStep):
    """Encoder step to handle variable number of features.

    Transforms the input to a fixed number of features by appending zeros.
    Also normalizes the input by the number of used features to keep the variance
    of the input constant, even when zeros are appended.
    """

    def __init__(
        self,
        num_features: int,
        normalize_by_used_features: bool = True,
        normalize_by_sqrt: bool = True,
        **kwargs: Any,
    ):
        """Initialize the VariableNumFeaturesEncoderStep.

        Args:
            num_features: The number of features to transform the input to.
            normalize_by_used_features: Whether to normalize by the number of used features.
            normalize_by_sqrt: Legacy option to normalize by sqrt instead of the number of used features.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_by_used_features = normalize_by_used_features
        self.num_features = num_features
        self.normalize_by_sqrt = normalize_by_sqrt
        self.number_of_used_features_ = None

    def _fit(self, x: torch.Tensor, **kwargs: Any) -> None:
        """Compute the number of used features on the training set.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).
        """
        sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)
        self.number_of_used_features_ = torch.clip(
            sel.sum(-1).unsqueeze(-1),
            min=1,
        ).cpu()

    def _transform(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Transform the input tensor to have a fixed number of features.

        Args:
            x: The input tensor of shape (seq_len, batch_size, num_features_old).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor of shape (seq_len, batch_size, num_features).
        """
        if x.shape[2] == 0:
            return torch.zeros(
                x.shape[0],
                x.shape[1],
                self.num_features,
                device=x.device,
                dtype=x.dtype,
            )
        if self.normalize_by_used_features:
            if self.normalize_by_sqrt:
                # Verified that this gives indeed unit variance with appended zeros
                x = x * torch.sqrt(
                    self.num_features / self.number_of_used_features_.to(x.device),
                )
            else:
                x = x * (self.num_features / self.number_of_used_features_.to(x.device))

        zeros_appended = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, zeros_appended], -1)
        return (x,)


class InputNormalizationEncoderStep(SeqEncStep):
    """Encoder step to normalize the input in different ways.

    Can be used to normalize the input to a ranking, remove outliers,
    or normalize the input to have unit variance.
    """

    def __init__(
        self,
        normalize_on_train_only: bool,
        normalize_to_ranking: bool,
        normalize_x: bool,
        remove_outliers: bool,
        remove_outliers_sigma: float = 4.0,
        seed: int = 0,
        **kwargs: Any,
    ):
        """Initialize the InputNormalizationEncoderStep.

        Args:
            normalize_on_train_only: Whether to compute normalization only on the training set.
            normalize_to_ranking: Whether to normalize the input to a ranking.
            normalize_x: Whether to normalize the input to have unit variance.
            remove_outliers: Whether to remove outliers from the input.
            remove_outliers_sigma: The number of standard deviations to use for outlier removal.
            seed: Random seed for reproducibility.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only
        self.normalize_to_ranking = normalize_to_ranking
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers
        self.remove_outliers_sigma = remove_outliers_sigma
        self.seed = seed
        self.reset_seed()
        self.lower_for_outlier_removal = None
        self.upper_for_outlier_removal = None
        self.mean_for_normalization = None
        self.std_for_normalization = None

    def reset_seed(self) -> None:
        """Reset the random seed."""

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
        """Compute the normalization statistics on the training set.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).
        """
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1
        if self.remove_outliers and not self.normalize_to_ranking:
            (
                x,
                (
                    self.lower_for_outlier_removal,
                    self.upper_for_outlier_removal,
                ),
            ) = remove_outliers(
                x,
                normalize_positions=normalize_position,
                n_sigma=self.remove_outliers_sigma,
            )

        if self.normalize_x:
            (
                x,
                (
                    self.mean_for_normalization,
                    self.std_for_normalization,
                ),
            ) = normalize_data(
                x,
                normalize_positions=normalize_position,
                return_scaling=True,
            )

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        """Normalize the input tensor.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the normalized tensor.
        """
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        if self.normalize_to_ranking:
            raise AssertionError(
                "Not implemented currently as it was not used in a long time and hard to move out the state.",
            )
            x = to_ranking_low_mem(x)

        if self.remove_outliers:
            assert self.remove_outliers_sigma > 1.0, (
                "remove_outliers_sigma must be > 1.0"
            )

            x, _ = remove_outliers(
                x,
                normalize_positions=normalize_position,
                lower=self.lower_for_outlier_removal,
                upper=self.upper_for_outlier_removal,
                n_sigma=self.remove_outliers_sigma,
            )

        if self.normalize_x:
            x = normalize_data(
                x,
                normalize_positions=normalize_position,
                mean=self.mean_for_normalization,
                std=self.std_for_normalization,
            )

        return (x,)


class FrequencyFeatureEncoderStep(SeqEncStep):
    """Encoder step to add frequency-based features to the input."""

    def __init__(
        self,
        num_features: int,
        num_frequencies: int,
        freq_power_base: float = 2.0,
        max_wave_length: float = 4.0,
        **kwargs: Any,
    ):
        """Initialize the FrequencyFeatureEncoderStep.

        Args:
            num_features: The number of input features.
            num_frequencies: The number of frequencies to add (both sin and cos).
            freq_power_base:
                The base of the frequencies.
                Frequencies will be `freq_power_base`^i for i in range(num_frequencies).
            max_wave_length: The maximum wave length.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.num_frequencies = num_frequencies
        self.num_features = num_features
        self.num_features_out = num_features + 2 * num_frequencies * num_features
        self.freq_power_base = freq_power_base
        # We add frequencies with a factor of freq_power_base
        wave_lengths = torch.tensor(
            [freq_power_base**i for i in range(num_frequencies)],
            dtype=torch.float,
        )
        wave_lengths = wave_lengths / wave_lengths[-1] * max_wave_length
        # After this adaption, the last (highest) wavelength is max_wave_length
        self.register_buffer("wave_lengths", wave_lengths)

    def _fit(
        self,
        x: torch.Tensor,
        single_eval_pos: int | None = None,
        categorical_inds: list[int] | None = None,
    ):
        """Fit the encoder step. Does nothing for FrequencyFeatureEncoderStep."""

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int | None = None,
        categorical_inds: list[int] | None = None,
    ):
        """Add frequency-based features to the input tensor.

        Args:
            x: The input tensor of shape (seq_len, batch_size, num_features).
            single_eval_pos: The position to use for single evaluation. Not used.
            categorical_inds: The indices of categorical features. Not used.

        Returns:
            A tuple containing the transformed tensor of shape
            `(seq_len, batch_size, num_features + 2 * num_frequencies * num_features)`.
        """
        extended = x[..., None] / self.wave_lengths[None, None, None, :] * 2 * torch.pi
        new_features = torch.cat(
            (x[..., None], torch.sin(extended), torch.cos(extended)),
            -1,
        )
        new_features = new_features.reshape(*x.shape[:-1], -1)
        return (new_features,)


class CategoricalInputEncoderPerFeatureEncoderStep(SeqEncStep):
    """Expects input of size 1."""

    def __init__(
        self,
        num_features,
        emsize,
        base_encoder,
        num_embs: int = 1_000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert num_features == 1
        self.num_features = num_features
        self.emsize = emsize
        self.num_embs = num_embs
        self.embedding = nn.Embedding(num_embs, emsize)
        self.base_encoder = base_encoder

    def _fit(self, x, single_eval_pos: int, categorical_inds: list[int]):
        pass

    def _transform(
        self,
        x,
        single_eval_pos: int,
        categorical_inds: list[int],
    ):
        if categorical_inds is None:
            is_categorical = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
        else:
            assert all(ci in ([0], []) for ci in categorical_inds), categorical_inds
            is_categorical = torch.tensor(
                [ci == [0] for ci in categorical_inds],
                device=x.device,
            )

        if is_categorical.any():
            lx = x[:, is_categorical]
            nan_mask = torch.isnan(lx) | torch.isinf(lx)
            lx = lx.long().clamp(0, self.num_embs - 2)
            lx[nan_mask] = self.num_embs - 1
            categorical_embs = self.embedding(lx.squeeze(-1))
        else:
            categorical_embs = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)

        if (~is_categorical).any():
            lx = x[:, ~is_categorical]
            continuous_embs = self.base_encoder(lx, single_eval_pos=single_eval_pos)[0]
        else:
            continuous_embs = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)

        # return (torch.cat((continuous_embs, categorical_embs), dim=1),)
        # above is wrong as we need to preserve order in the batch dimension
        embs = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.emsize,
            device=x.device,
            dtype=torch.float,
        )
        embs[:, is_categorical] = categorical_embs.float()
        embs[:, ~is_categorical] = continuous_embs.float()
        return (embs,)


class StyleEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size):
        super().__init__()
        self.em_size = em_size
        self.embedding = nn.Linear(num_hyperparameters, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters)


def get_linear_encoder_generator(in_keys):
    def get_linear_encoder(num_features, emsize):
        return SequentialEncoder(
            LinearInputEncoderStep(
                num_features,
                emsize,
                in_keys=in_keys,
                out_keys=["output"],
            ),
            output_key="output",
        )

    return get_linear_encoder


##### TARGET ENCODERS #####


class MulticlassClassificationTargetEncoder(SeqEncStep):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.unique_ys_ = None

    def _fit(self, y: torch.Tensor, single_eval_pos: int, **kwargs: Any):
        assert len(y.shape) == 3 and (y.shape[-1] == 1), "y must be of shape (T, B, 1)"
        self.unique_ys_ = [
            torch.unique(y[:single_eval_pos, b_i]) for b_i in range(y.shape[1])
        ]

    @staticmethod
    def flatten_targets(y: torch.Tensor, unique_ys: torch.Tensor | None = None):
        if unique_ys is None:
            unique_ys = torch.unique(y)
        return (y.unsqueeze(-1) > unique_ys).sum(axis=-1)

    def _transform(self, y: torch.Tensor, single_eval_pos: int | None = None):
        assert len(y.shape) == 3 and (y.shape[-1] == 1), "y must be of shape (T, B, 1)"
        assert not (y.isnan().any() and self.training), (
            "NaNs are not allowed in the target at this point during training (set to model.eval() if not in training)"
        )
        y_new = y.clone()
        for B in range(y.shape[1]):
            y_new[:, B, :] = self.flatten_targets(y[:, B, :], self.unique_ys_[B])
        return (y_new,)


class Activation(Enum):
    """Enum for activation functions."""

    GELU = 1
    RELU = 2


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron (MLP) module.

    This module consists of two linear layers with an activation function in between.
    It supports various configurations such as the hidden size, activation function,
    initializing the output to zero, and recomputing the forward pass during
    backpropagation.

    Args:
        size: The input and output size of the MLP.
        hidden_size: The size of the hidden layer.
        activation:
            The activation function to use. Can be either an Activation enum or
            a string representing the activation name.
        device: The device to use for the linear layers.
        dtype: The data type to use for the linear layers.
        initialize_output_to_zero:
            Whether to initialize the output layer weights
            to zero. Default is False.
        recompute:
            Whether to recompute the forward pass during backpropagation.
            This can save memory but increase computation time. Default is False.

    Attributes:
        linear1: The first linear layer.
        linear2: The second linear layer.
        activation: The activation function to use.

    Example:
        >>> mlp = MLP(size=128, hidden_size=256, activation='gelu', device='cuda')
        >>> x = torch.randn(32, 128, device='cuda', dtype=torch.float32)
        >>> output = mlp(x)
    """

    linear1: torch.nn.Linear
    linear2: torch.nn.Linear
    activation: Activation

    def __init__(
        self,
        size: int,
        hidden_size: int,
        activation: Activation | str,
        *,
        device: torch.device | None,
        dtype: torch.dtype | None,
        initialize_output_to_zero: bool = False,
        recompute: bool = False,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(
            size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.linear2 = torch.nn.Linear(
            hidden_size,
            size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if isinstance(activation, str):
            activation = Activation[activation.upper()]
        self.activation = activation
        if initialize_output_to_zero:
            torch.nn.init.zeros_(self.linear2.weight)
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore

    @support_save_peak_mem_factor  # type: ignore
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        if self.activation is Activation.GELU:
            x = torch.nn.functional.gelu(x)
        elif self.activation is Activation.RELU:
            x = torch.nn.functional.relu(x)
        else:
            raise NotImplementedError(
                f"Activation Function {self.activation} is not implemented.",
            )
        return self.linear2(x)

    def forward(
        self,
        x: torch.Tensor,
        *,
        add_input: bool = False,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
    ) -> torch.Tensor:
        """Performs the forward pass of the MLP.

        Args:
            x: The input tensor.
            add_input: Whether to add input to the output. Default is False.
            allow_inplace:
                Indicates that 'x' is not used after the call and
                its buffer can be reused for the output. The operation is not
                guaranteed to be inplace. Default is False.
            save_peak_mem_factor:
                If provided, enables a
                memory-saving technique that reduces peak memory usage during the
                forward pass. This requires 'add_input' and 'allow_inplace' to be True.
                See the documentation of the decorator 'support_save_peak_mem_factor'
                for details. Default is None.
        """
        input_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self._compute(
            x,
            add_input=add_input,
            allow_inplace=allow_inplace,
            save_peak_mem_factor=save_peak_mem_factor,
        )
        return x.reshape(input_shape)


try:
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_func,
        flash_attn_unpadded_kvpacked_func,
        flash_attn_unpadded_qkvpacked_func,
    )

    HAVE_FLASH_ATTN = True
except (ModuleNotFoundError, ImportError):
    HAVE_FLASH_ATTN = False


class MultiHeadAttention(torch.nn.Module):
    _input_size: int
    _output_size: int
    _nhead: int
    _nhead_kv: int
    _d_k: int
    _d_v: int
    _share_kv_across_n_heads: int
    dropout_p: float | None
    softmax_scale: float | None
    _k_cache: torch.Tensor | None
    _v_cache: torch.Tensor | None
    _kv_cache: torch.Tensor | None
    _w_q: torch.nn.Parameter | None
    _w_k: torch.nn.Parameter | None
    _w_v: torch.nn.Parameter | None
    _w_kv: torch.nn.Parameter | None
    _w_qkv: torch.nn.Parameter | None
    _w_out: torch.nn.Parameter

    @property
    def w_q(self) -> torch.nn.Parameter | None:
        return self._w_q

    @property
    def w_k(self) -> torch.nn.Parameter | None:
        return self._w_k

    @property
    def w_v(self) -> torch.nn.Parameter | None:
        return self._w_v

    @property
    def w_qkv(self) -> torch.nn.Parameter | None:
        return self._w_qkv

    @property
    def w_kv(self) -> torch.nn.Parameter | None:
        return self._w_kv

    @property
    def w_out(self) -> torch.nn.Parameter:
        return self._w_out

    @property
    def has_cached_kv(self) -> bool:
        assert (self._k_cache is None) == (self._v_cache is None)
        assert self._kv_cache is None or (
            self._k_cache is None and self._v_cache is None
        )
        return (
            self._k_cache is not None and self._v_cache is not None
        ) or self._kv_cache is not None

    def empty_kv_cache(self):
        self._k_cache = None
        self._v_cache = None
        self._kv_cache = None

    def set_parameters(
        self,
        w_out: torch.nn.Parameter,
        w_q: torch.nn.Parameter | None = None,
        w_k: torch.nn.Parameter | None = None,
        w_v: torch.nn.Parameter | None = None,
        w_kv: torch.nn.Parameter | None = None,
        w_qkv: torch.nn.Parameter | None = None,
        precomputed_k: torch.Tensor | None = None,
        precomputed_v: torch.Tensor | None = None,
        precomputed_kv: torch.Tensor | None = None,
    ):
        assert (precomputed_k is None) == (precomputed_v is None)
        assert (precomputed_kv is None) or (precomputed_k is None)
        assert (precomputed_kv is None and precomputed_k is None) != (
            w_qkv is None and w_kv is None and w_k is None and w_v is None
        )
        assert (w_qkv is None) != (w_q is None)
        assert (w_qkv is None) or (w_kv is None and w_k is None and w_v is None)
        assert w_kv is None or (w_k is None and w_v is None)
        assert (w_k is None) == (w_v is None)

        def assert_tensor_shape(
            tensor: torch.Tensor | None,
            expected_shape: list[int | None],
        ):
            if tensor is None:
                return
            actual_shape = tensor.size()
            err = f"Tensor {actual_shape=} does not match {expected_shape=}."
            assert len(actual_shape) == len(expected_shape), err
            for actual_dim, expected_dim in zip(actual_shape, expected_shape):
                if expected_dim is not None:
                    assert actual_dim == expected_dim, err

        assert_tensor_shape(precomputed_k, [None, None, self._nhead_kv, self._d_k])
        assert_tensor_shape(precomputed_v, [None, None, self._nhead_kv, self._d_v])
        assert_tensor_shape(precomputed_kv, [None, None, 2, self._nhead_kv, self._d_k])
        assert_tensor_shape(
            w_q,
            [
                1 + int(bool(self.two_sets_of_queries)),
                self._nhead,
                self._d_k,
                self._input_size,
            ],
        )
        assert_tensor_shape(w_k, [self._nhead_kv, self._d_k, self._input_size])
        assert_tensor_shape(w_v, [self._nhead_kv, self._d_v, self._input_size])
        assert_tensor_shape(w_kv, [2, self._nhead_kv, self._d_k, self._input_size])
        assert_tensor_shape(w_qkv, [3, self._nhead, self._d_k, self._input_size])
        assert_tensor_shape(w_out, [self._nhead, self._d_v, self._output_size])

        self.register_parameter("_w_out", w_out)
        self.register_parameter("_w_q", w_q)
        self.register_parameter("_w_k", w_k)
        self.register_parameter("_w_v", w_v)
        self.register_parameter("_w_kv", w_kv)
        self.register_parameter("_w_qkv", w_qkv)

        self.register_buffer("_k_cache", precomputed_k)
        self.register_buffer("_v_cache", precomputed_v)
        self.register_buffer("_kv_cache", precomputed_kv)

    def newly_initialized_input_weight(
        self,
        dims: list[int],
        nhead: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> torch.nn.Parameter:
        assert 3 <= len(dims) <= 4  # ([stack,] nhead_, d, input_size)
        w = torch.nn.Parameter(torch.empty(*dims, device=device, dtype=dtype))
        d, input_size = dims[-2:]
        std = math.sqrt(2.0 / float(nhead * d + input_size)) * self.init_gain
        a = math.sqrt(3.0) * std
        torch.nn.init.uniform_(w, -a, a)
        return w

    def __init__(  # noqa: PLR0913
        self,
        *,
        input_size: int,
        output_size: int,
        d_k: int,
        d_v: int,
        nhead: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
        share_kv_across_n_heads: int = 1,
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
        initialize_output_to_zero: bool = False,
        precomputed_k: torch.Tensor | None = None,
        precomputed_v: torch.Tensor | None = None,
        precomputed_kv: torch.Tensor | None = None,
        recompute: bool = False,
        init_gain: float = 1.0,
        two_sets_of_queries: bool = False,
    ):
        super().__init__()
        assert nhead % share_kv_across_n_heads == 0
        self._input_size = input_size
        self._output_size = output_size
        self._d_k = d_k
        self._d_v = d_v
        self._nhead = nhead
        self._nhead_kv = nhead // share_kv_across_n_heads
        self._device = device
        self._dtype = dtype
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.recompute = recompute
        self.init_gain = init_gain
        self.two_sets_of_queries = two_sets_of_queries

        w_out = torch.nn.Parameter(
            torch.empty(nhead, d_v, output_size, device=device, dtype=dtype),
        )
        if initialize_output_to_zero:
            torch.nn.init.zeros_(w_out)
        else:
            torch.nn.init.xavier_uniform_(w_out)

        assert precomputed_k is None == precomputed_v is None
        has_precomputed_kv = precomputed_kv is not None or precomputed_k is not None
        w_q = None
        w_k = None
        w_v = None
        w_kv = None
        w_qkv = None
        if (
            d_k == d_v
            and self._nhead == self._nhead_kv
            and not has_precomputed_kv
            and not two_sets_of_queries
        ):
            w_qkv = self.newly_initialized_input_weight(
                [3, self._nhead, self._d_k, self._input_size],
                nhead=self._nhead,
                device=device,
                dtype=dtype,
            )
        else:
            w_q = self.newly_initialized_input_weight(
                [
                    1 + int(bool(two_sets_of_queries)),
                    self._nhead,
                    self._d_k,
                    self._input_size,
                ],
                nhead=self._nhead,
                device=device,
                dtype=dtype,
            )
            if not has_precomputed_kv:
                if d_k == d_v:
                    w_kv = self.newly_initialized_input_weight(
                        [2, self._nhead_kv, self._d_k, self._input_size],
                        nhead=self._nhead,
                        device=device,
                        dtype=dtype,
                    )
                else:
                    w_k = self.newly_initialized_input_weight(
                        [self._nhead_kv, self._d_k, self._input_size],
                        nhead=self._nhead,
                        device=device,
                        dtype=dtype,
                    )
                    w_v = self.newly_initialized_input_weight(
                        [self._nhead_kv, self._d_v, self._input_size],
                        nhead=self._nhead,
                        device=device,
                        dtype=dtype,
                    )
        self.set_parameters(
            w_out,
            w_q,
            w_k,
            w_v,
            w_kv,
            w_qkv,
            precomputed_k,
            precomputed_v,
            precomputed_kv,
        )
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        *,
        cache_kv: bool = False,
        add_input: bool = False,
        # Indicates that 'x' is not used after the call and its buffer can be reused
        # for the output. The operation is not guaranteed to be inplace.
        allow_inplace: bool = False,
        # This requires 'add_input' and 'allow_inplace'. See the documentation of
        # the decorator 'support_save_peak_mem_factor' for details.
        save_peak_mem_factor: int | None = None,
        reuse_first_head_kv: bool = False,
        only_cache_first_head_kv: bool = False,
        use_cached_kv: bool = False,
        use_second_set_of_queries: bool = False,
    ):
        """X is the current hidden and has a shape of [batch, ..., seq_len, input_size].
        If keys and values are present in the cache and 'freeze_kv' is not set, they
        are obtained from there and 'x_kv' has to be None.
        Else, if 'x_kv' is not None, keys and values are obtained by applying the
        respective linear transformations to 'x_kv'.
        Else, keys and values are attained by applying the respective linear
        transformations to 'x' (self attention).
        """
        assert not (cache_kv and use_cached_kv), (
            "Cannot cache and use cached keys and values at the same time."
        )
        if use_second_set_of_queries:
            assert self.two_sets_of_queries, (
                "Two sets of queries are not supported."
                "Please set 'two_sets_of_queries' to True."
            )
        assert not x.requires_grad or (not self.has_cached_kv and not cache_kv), (
            "Saving keys and values is only supported during inference."
        )
        x, x_kv, x_shape_after_transpose = self._rearrange_inputs_to_flat_batch(x, x_kv)

        nhead_kv = 1 if reuse_first_head_kv else self._nhead_kv

        if cache_kv:
            # Reset cache first so memory is freed before new cache is allocated.
            self._k_cache = self._v_cache = self._kv_cache = None

            if x_kv is not None:
                batch_size, seqlen_kv = x_kv.shape[:2]
            else:
                batch_size, seqlen_kv = x.shape[:2]

            # TODO: handling of device and dtype.
            if self._w_kv is not None or self._w_qkv is not None:
                self._kv_cache = torch.empty(
                    batch_size,
                    seqlen_kv,
                    2,
                    1 if only_cache_first_head_kv else nhead_kv,
                    self._d_k,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                self._k_cache = torch.empty(
                    batch_size,
                    seqlen_kv,
                    nhead_kv,
                    self._d_k,
                    device=x.device,
                    dtype=x.dtype,
                )
                self._v_cache = torch.empty(
                    batch_size,
                    seqlen_kv,
                    nhead_kv,
                    self._d_v,
                    device=x.device,
                    dtype=x.dtype,
                )

        output: torch.Tensor = self._compute(
            x,
            x_kv,
            self._k_cache,
            self._v_cache,
            self._kv_cache,
            cache_kv=cache_kv,
            use_cached_kv=use_cached_kv,
            add_input=add_input,
            allow_inplace=allow_inplace,
            save_peak_mem_factor=save_peak_mem_factor,
            reuse_first_head_kv=reuse_first_head_kv,
            use_second_set_of_queries=use_second_set_of_queries,
        )
        return output.reshape(x_shape_after_transpose[:-1] + output.shape[-1:])

    def compute_qkv(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        *,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        use_second_set_of_queries: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        assert not (cache_kv and use_cached_kv), (
            "You cannot both cache new KV and use the cached KV at once."
        )
        if reuse_first_head_kv:
            assert x is not x_kv, (
                "x and x_kv must be different tensors. That means reuse_first_head_kv"
                "is not compatible with self attention only cross attention."
            )
        if x_kv is None:
            x_kv = x

        k = v = kv = None
        if use_cached_kv:
            assert self.has_cached_kv, (
                "You try to use cached keys and values but the cache is empty."
            )
            k = k_cache
            v = v_cache
            kv = kv_cache

        assert (k is None) == (v is None)

        if use_second_set_of_queries:
            assert self._w_qkv is None, (
                "Two sets of queries are not supported with custom q weights,"
                " set two_sets_of_queries to True."
            )

        if self._w_qkv is None:
            w_q, w_kv = self._w_q[int(bool(use_second_set_of_queries))], self._w_kv
        else:
            w_q, w_kv = self._w_qkv[0], self._w_qkv[1:]

        if (
            self._w_qkv is not None
            and x is x_kv
            and kv is None
            and k is None
            and v is None
        ):
            qkv = torch.einsum("... s, j h d s -> ... j h d", x, self._w_qkv)
            q = None
        else:
            qkv = None
            q = torch.einsum("... s, h d s -> ... h d", x, w_q)

        if kv is None and k is None and v is None and qkv is None:
            if w_kv is not None:
                if reuse_first_head_kv:
                    orig_num_heads = w_kv.shape[1]
                    w_kv = w_kv[:, :1]
                kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, w_kv)
                if reuse_first_head_kv:
                    expand_shape = [-1 for _ in kv.shape]
                    expand_shape[-2] = orig_num_heads
                    kv = kv.expand(*expand_shape)
            else:
                w_k = self._w_k
                w_v = self._w_v
                if reuse_first_head_kv:
                    orig_num_heads = w_k.shape[0]
                    w_k = w_k[:1]
                    w_v = w_v[:1]
                k = torch.einsum("... s, h d s -> ... h d", x_kv, w_k)
                v = torch.einsum("... s, h d s -> ... h d", x_kv, w_v)
                if reuse_first_head_kv:
                    expand_shape = [-1 for _ in k.shape]
                    expand_shape[-2] = orig_num_heads
                    k = k.expand(*expand_shape)
                    v = v.expand(*expand_shape)

        if cache_kv:
            if k_cache is not None:
                k_cache[:] = k
            if v_cache is not None:
                v_cache[:] = v
            if kv_cache is not None:
                if kv_cache.shape[-2] == 1:
                    # we are in the case where only the first head kv is cached
                    # that is the case when we only neeed that for inference
                    kv_cache[:] = kv[..., :1, :]
                else:
                    kv_cache[:] = kv

        return q, k, v, kv, qkv

    @support_save_peak_mem_factor  # type: ignore
    def _compute(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        *,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        use_second_set_of_queries: bool,
    ) -> torch.Tensor:
        """Attention computation.
        Called by 'forward', potentially on shards, once shapes have been normalized.
        """
        q, k, v, kv, qkv = self.compute_qkv(
            x,
            x_kv,
            k_cache,
            v_cache,
            kv_cache,
            cache_kv=cache_kv,
            use_cached_kv=use_cached_kv,
            reuse_first_head_kv=reuse_first_head_kv,
            use_second_set_of_queries=use_second_set_of_queries,
        )
        attention_head_outputs = MultiHeadAttention.compute_attention_heads(
            q,
            k,
            v,
            kv,
            qkv,
            self.dropout_p,
            self.softmax_scale,
        )
        return torch.einsum(
            "... h d, h d s -> ... s",
            attention_head_outputs,
            self._w_out,
        )

    def _rearrange_inputs_to_flat_batch(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Size]:
        # TODO: This presumably creates potential memory overhead not captured
        # by save_peak_mem_factor.
        x_shape_after_transpose = x.shape
        if x_kv is not None:
            assert x.shape[:-2] == x_kv.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])
        if x_kv is not None:
            x_kv = x_kv.reshape(-1, *x_kv.shape[-2:])
        return x, x_kv, x_shape_after_transpose

    @staticmethod
    def broadcast_kv_across_heads(
        kv: torch.Tensor,
        share_kv_across_n_heads: int,
    ) -> torch.Tensor:
        nhead, d = kv.shape[-2:]
        kv = kv[..., None, :].expand(
            *([-1] * (kv.dim() - 1)),
            share_kv_across_n_heads,
            -1,
        )
        return kv.reshape(*kv.shape[:-3], nhead * share_kv_across_n_heads, d)

    @staticmethod
    def compute_attention_heads(  # noqa: C901, PLR0912
        q: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        kv: torch.Tensor | None,
        qkv: torch.Tensor | None,
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        assert (k is None) == (v is None)
        assert sum([qkv is None, kv is None, k is None and v is None]) == 2
        assert (qkv is None) != (q is None)

        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None:
            k, v = kv.unbind(dim=-3)

        assert q is not None
        assert k is not None
        assert v is not None

        batch_size, seqlen_q, nhead, d_k = q.shape
        _, seqlen_kv, nhead_kv, d_v = v.shape
        share_kv_across_n_heads = nhead // nhead_kv
        if dropout_p is None:
            dropout_p = 0.0  # TODO: necessary?

        use_flash_attention = (
            HAVE_FLASH_ATTN
            and torch.cuda.is_available()
            and q.dtype == k.dtype == v.dtype == torch.float16
        )

        # this string comparison is reliable, as it does not compare to a subversion
        TORCH_2_ATTENTION_POSSIBLE = (
            torch.__version__ >= "2" and torch.cuda.is_available()
        )
        USE_TORCH_2_GQA = False
        if TORCH_2_ATTENTION_POSSIBLE:
            # check whether torch.nn.functional.scaled_dot_product_attention has a
            # kwarg enable_gqa
            # Check if enable_gqa is supported by trying to call the function with
            # the parameter
            try:
                _ = torch.nn.functional.scaled_dot_product_attention(
                    torch.empty(1, 1, 1, 1),
                    torch.empty(1, 1, 1, 1),
                    torch.empty(1, 1, 1, 1),
                    enable_gqa=True,
                )
                TORCH_2_SUPPORTS_GQ = True
            except (TypeError, RuntimeError):
                TORCH_2_SUPPORTS_GQ = False

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                capability = torch.cuda.get_device_capability(device)
                nvidia_compute_capability = f"{capability[0]}.{capability[1]}"
            else:
                nvidia_compute_capability = None
            USE_TORCH_2_GQA = nvidia_compute_capability >= "8" and TORCH_2_SUPPORTS_GQ

            # TODO: add logging for something like this
            # if use_flash_attention and USE_TORCH_2_GQA:
            # print("Using FlashAttention might be slower than torch's implementation,
            # try setting `tabpfn.model.multi_head_attention.HAVE_FLASH_ATTN=False`.")

            # print(f"USE_TORCH_2_GQA: {USE_TORCH_2_GQA}, nvidia_compute_capability:
            # {nvidia_compute_capability}, TORCH_2_SUPPORTS_GQ: {TORCH_2_SUPPORTS_GQ}")

        if use_flash_attention:

            def get_seqlen_cumsums(
                batch_size: int,
                seqlen: int,
                device: torch.device,
            ) -> torch.Tensor:
                return torch.arange(
                    0,
                    (batch_size + 1) * seqlen,
                    step=seqlen,
                    dtype=torch.int32,
                    device=device,
                )

            if qkv is not None:
                attention_head_outputs = flash_attn_unpadded_qkvpacked_func(  # type: ignore
                    qkv.reshape(batch_size * seqlen_q, 3, nhead, d_k),
                    get_seqlen_cumsums(batch_size, seqlen_q, qkv.device),
                    seqlen_q,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,  # defaults to 1/sqrt(d_k) if None
                    causal=False,
                    return_attn_probs=False,
                    deterministic=False,
                )
            elif kv is not None:
                kv = MultiHeadAttention.broadcast_kv_across_heads(
                    kv,
                    share_kv_across_n_heads,
                )
                attention_head_outputs = flash_attn_unpadded_kvpacked_func(  # type: ignore
                    q.reshape(batch_size * seqlen_q, nhead, d_k),
                    kv.reshape(batch_size * seqlen_kv, 2, nhead, d_k),
                    get_seqlen_cumsums(batch_size, seqlen_q, q.device),
                    get_seqlen_cumsums(batch_size, seqlen_kv, kv.device),
                    seqlen_q,
                    seqlen_kv,
                    dropout_p=dropout_p,
                    causal=False,
                    return_attn_probs=False,
                    deterministic=False,
                )
            else:
                assert d_k <= d_v, (
                    "This requirement is here for safety but not strictly necessary."
                    "Needs testing/coding to remove."
                )
                if d_k < d_v:
                    k = torch.nn.functional.pad(k, d_v - d_k)
                    q = torch.nn.functional.pad(v, d_v - d_k)
                    d_k_ = d_v
                k = MultiHeadAttention.broadcast_kv_across_heads(
                    k,
                    share_kv_across_n_heads,
                )
                v = MultiHeadAttention.broadcast_kv_across_heads(
                    v,
                    share_kv_across_n_heads,
                )
                attention_head_outputs = flash_attn_unpadded_func(  # type: ignore
                    q.reshape(batch_size * seqlen_q, nhead, d_k_),  # type: ignore
                    k.reshape(batch_size * seqlen_kv, nhead, d_k_),  # type: ignore
                    v.reshape(batch_size * seqlen_kv, nhead, d_v),
                    get_seqlen_cumsums(batch_size, seqlen_q, q.device),
                    get_seqlen_cumsums(batch_size, seqlen_kv, k.device),
                    seqlen_q,
                    seqlen_kv,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=False,
                    return_attn_probs=False,
                    deterministic=False,
                )
        elif TORCH_2_ATTENTION_POSSIBLE:
            extra_inputs = {}
            if softmax_scale is not None:
                extra_inputs["scale"] = (
                    softmax_scale  # defaults to 1/sqrt(d_k) if None or not provided
                )
            if not USE_TORCH_2_GQA:
                k = MultiHeadAttention.broadcast_kv_across_heads(
                    k,
                    share_kv_across_n_heads,
                )
                v = MultiHeadAttention.broadcast_kv_across_heads(
                    v,
                    share_kv_across_n_heads,
                )
            else:
                extra_inputs["enable_gqa"] = True
            attention_head_outputs = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout_p,
                **extra_inputs,
            )
            attention_head_outputs = attention_head_outputs.transpose(1, 2)
        else:
            k = MultiHeadAttention.broadcast_kv_across_heads(k, share_kv_across_n_heads)
            v = MultiHeadAttention.broadcast_kv_across_heads(v, share_kv_across_n_heads)
            logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
            logits *= (
                torch.sqrt(torch.tensor(1.0 / d_k)).to(k.device)
                if softmax_scale is None
                else softmax_scale
            )
            ps = torch.softmax(logits, dim=2)
            ps = torch.dropout(ps, dropout_p, train=True)
            attention_head_outputs = torch.einsum("b q k h, b k h d -> b q h d", ps, v)

        return attention_head_outputs.reshape(
            batch_size,
            seqlen_q,
            nhead,
            d_v,
        )

    @staticmethod
    def convert_torch_nn_multihead_attention_state_dict(
        state_dict: dict,
        nhead: int,
        *,
        disable_stacked_w_qkv: bool = False,
    ) -> dict:
        in_proj_weight = state_dict["in_proj_weight"]
        out_proj_weight = state_dict["out_proj.weight"]

        embed_dim = in_proj_weight.shape[1]
        assert embed_dim % nhead == 0
        assert in_proj_weight.shape[0] == 3 * embed_dim
        assert out_proj_weight.shape == (embed_dim, embed_dim)
        in_proj_weight = in_proj_weight.reshape(3, nhead, -1, embed_dim)

        state_dict = {}
        if disable_stacked_w_qkv:
            state_dict["_w_q"], state_dict["_w_kv"] = torch.split(
                in_proj_weight,
                [1, 2],
            )
            state_dict["_w_q"] = state_dict["_w_q"].squeeze(0)
        else:
            state_dict["_w_qkv"] = in_proj_weight
        state_dict["_w_out"] = out_proj_weight.T.reshape(nhead, -1, embed_dim)
        return state_dict


HIDDEN_SIZE_LIMIT = 512
MLP_SAVE_PEAK_MEM_FACTOR = 32


class LayerNorm(torch.nn.LayerNorm):
    """Custom LayerNorm module that supports saving peak memory factor.

    This module extends the PyTorch LayerNorm implementation to handle FP16 inputs
    efficiently and support saving peak memory factor.

    Args:
        *args: Positional arguments passed to the base LayerNorm class.
        **kwargs: Keyword arguments passed to the base LayerNorm class.
    """

    # TODO: Not sure why this needs to be wrapped with functools.wraps
    @functools.wraps(torch.nn.LayerNorm.__init__)  # type: ignore
    def __init__(self, *args: Any, **kwargs: Any):  # type: ignore
        super().__init__(*args, **kwargs)

    @support_save_peak_mem_factor  # type: ignore
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the layer normalization.

        If the input is FP16 and the normalized shape is less than 512, the computation
        is forced to be in FP16 to improve performance.

        Args:
            x: The input tensor.

        Returns:
            The layer normalized tensor.
        """
        # this if statement has the following function:
        # torch.amp.autocast wants to run layer_norm in fp32
        # but that has very bad effects for our performance (up to 2x slower)
        # thus we force fp16, if the input to the module is fp16, which is the case
        # if autocast is used.
        # WARNING: this could lead to instabilities for higher hidden sizes (> 512),
        # thus we only do this for smaller hidden sizes

        if x.dtype == torch.float16 and sum(self.normalized_shape) < HIDDEN_SIZE_LIMIT:
            with torch.amp.autocast("cuda" if x.is_cuda else "cpu", enabled=False):
                return super().forward(x)

        return super().forward(x)

    def forward(
        self,
        input: torch.Tensor,
        *,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
    ) -> torch.Tensor:
        """Perform layer normalization on the input tensor.

        Args:
            input: The input tensor.
            allow_inplace: Whether to allow in-place operations. Default is False.
            save_peak_mem_factor: The factor to save peak memory. Default is None.

        Returns:
            The layer normalized tensor.
        """
        x = input
        input_shape = x.shape

        x = x.reshape(-1, *self.normalized_shape)
        x = self._compute(
            x,
            allow_inplace=allow_inplace,
            save_peak_mem_factor=save_peak_mem_factor,
        )
        return x.reshape(input_shape)


class PerFeatureEncoderLayer(Module):
    """Transformer encoder layer that processes each feature block separately.

    This layer consists of multi-head attention between features, multi-head
    attention between items, and feedforward neural networks (MLPs).

    It supports various configurations and optimization options.

    Args:
        d_model: The dimensionality of the input and output embeddings.
        nhead: The number of attention heads.
        dim_feedforward:
            The dimensionality of the feedforward network.
            Default is None (2 * d_model).
        activation: The activation function to use in the MLPs.
        layer_norm_eps: The epsilon value for layer normalization.
        pre_norm:
            Whether to apply layer normalization before or after the attention
            and MLPs.
        device: The device to use for the layer parameters.
        dtype: The data type to use for the layer parameters.
        recompute_attn: Whether to recompute attention during backpropagation.
        second_mlp: Whether to include a second MLP in the layer.
        layer_norm_with_elementwise_affine:
            Whether to use elementwise affine parameters in layer normalization.
        zero_init: Whether to initialize the output of the MLPs to zero.
        save_peak_mem_factor:
            The factor to save peak memory, only effective with post-norm.
        attention_between_features: Whether to apply attention between feature blocks.
        multiquery_item_attention: Whether to use multiquery attention for items.
        multiquery_item_attention_for_test_set:
            Whether to use multiquery attention for the test set.
        attention_init_gain: The gain value for initializing attention parameters.
        d_k:
            The dimensionality of the query and key vectors.
            Default is (d_model // nhead).
        d_v:
            The dimensionality of the value vectors. Default is (d_model // nhead).
        precomputed_kv: Precomputed key-value pairs for attention.
    """

    __constants__: ClassVar = ["batch_first"]

    def __init__(  # noqa: PLR0913
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        recompute_attn: bool = False,
        second_mlp: bool = False,
        layer_norm_with_elementwise_affine: bool = False,
        zero_init: bool = False,
        save_peak_mem_factor: int | None = None,
        attention_between_features: bool = True,
        multiquery_item_attention: bool = False,
        multiquery_item_attention_for_test_set: bool = False,
        two_sets_of_queries: bool = False,
        attention_init_gain: float = 1.0,
        d_k: int | None = None,
        d_v: int | None = None,
        precomputed_kv: None | torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert d_model % nhead == 0 or (d_k is not None and d_v is not None)
        if multiquery_item_attention_for_test_set and multiquery_item_attention:
            raise ValueError(
                "Cannot use both multiquery_item_attention_for_test_set"
                "and multiquery_item_attention",
            )
        if two_sets_of_queries and not multiquery_item_attention_for_test_set:
            raise ValueError(
                "two_sets_of_queries requires multiquery_item_attention_for_test_set",
            )

        if d_k is None:
            d_k = d_model // nhead

        if d_v is None:
            d_v = d_model // nhead

        self.self_attn_between_features: MultiHeadAttention | None = None
        if attention_between_features:
            self.self_attn_between_features = MultiHeadAttention(
                input_size=d_model,
                output_size=d_model,
                d_k=d_k,
                d_v=d_v,
                nhead=nhead,
                device=device,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_attn,
                init_gain=attention_init_gain,
            )

        if isinstance(precomputed_kv, tuple):
            precomputed_k, precomputed_v = precomputed_kv
            precomputed_kv = None
        else:
            precomputed_k = precomputed_v = None

        self.self_attn_between_items = MultiHeadAttention(
            input_size=d_model,
            output_size=d_model,
            d_k=d_k,
            d_v=d_v,
            nhead=nhead,
            device=device,
            dtype=dtype,
            share_kv_across_n_heads=nhead if multiquery_item_attention else 1,
            initialize_output_to_zero=zero_init,
            recompute=recompute_attn,
            precomputed_k=precomputed_k,
            precomputed_v=precomputed_v,
            precomputed_kv=precomputed_kv,
            init_gain=attention_init_gain,
            two_sets_of_queries=(
                multiquery_item_attention_for_test_set and two_sets_of_queries
            ),
        )

        if dim_feedforward is None:
            dim_feedforward = 2 * d_model

        self.mlp = MLP(
            size=d_model,
            hidden_size=dim_feedforward,
            activation=activation,
            device=device,
            dtype=dtype,
            initialize_output_to_zero=zero_init,
            recompute=recompute_attn,
        )

        self.layer_norms = nn.ModuleList(
            [
                LayerNorm(
                    d_model,  # type: ignore
                    layer_norm_eps,
                    elementwise_affine=layer_norm_with_elementwise_affine,
                    **factory_kwargs,
                )
                for _ in range(4 if second_mlp else 3)
            ],
        )

        self.second_mlp: MLP | None = None
        if second_mlp:
            self.second_mlp = MLP(
                size=d_model,
                hidden_size=dim_feedforward,
                activation=activation,
                device=device,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_attn,
            )

        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.save_peak_mem_factor = save_peak_mem_factor
        self.multiquery_item_attention_for_test_set = (
            multiquery_item_attention_for_test_set
        )
        self.two_sets_of_queries = two_sets_of_queries

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("save_peak_mem_factor", None)
        super().__setstate__(state)

    def forward(  # noqa: C901
        self,
        state: Tensor,
        single_eval_pos: int | None = None,
        *,
        cache_trainset_representation: bool = False,
        att_src: Tensor | None = None,
    ) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            state:
                The transformer state passed as input to the layer of shape
                (batch_size, num_items, num_feature_blocks, d_model).
            single_eval_pos:
                The position from which on everything is treated as test
                set.
            cache_trainset_representation:
                Whether to cache the trainset representation.
                If single_eval_pos is set (> 0 and not None), create a cache of the
                trainset KV. This may require a lot of memory. Otherwise, use
                cached KV representations for inference.
            att_src:
                The tensor to attend to from the final layer of the encoder.
                It has a shape of
                (batch_size, num_train_items, num_feature_blocks, d_model).
                This does not work with multiquery_item_attention_for_test_set and
                cache_trainset_representation at this point.

        Returns:
            The transformer state passed through the encoder layer.
        """
        assert len(state.shape) == 4, (
            "src must be of shape (batch_size, num_items, num feature blocks, d_model)"
        )
        if single_eval_pos is None:
            single_eval_pos = 0

        save_peak_mem_factor = self.save_peak_mem_factor
        if cache_trainset_representation and not single_eval_pos:
            assert self.self_attn_between_items.has_cached_kv
            save_peak_mem_factor = None

        if att_src is not None:
            assert not self.multiquery_item_attention_for_test_set, (
                "Not implemented yet."
            )
            assert not cache_trainset_representation, "Not implemented yet."
            assert not single_eval_pos, (
                "single_eval_pos should not be set, as the train representation"
                " is in att_src"
            )

        if self.self_attn_between_features is None:
            assert not cache_trainset_representation, "Not implemented yet."
            assert state.shape[2] == 1, (
                f"One group architecture expects one feature group, "
                f"but got {state.shape[2]} feature groups."
            )

        def attn_between_features(x: torch.Tensor) -> torch.Tensor:
            assert self.self_attn_between_features is not None
            return self.self_attn_between_features(
                x,
                save_peak_mem_factor=save_peak_mem_factor,
                add_input=True,
                allow_inplace=True,
            )

        def attn_between_items(x: torch.Tensor) -> torch.Tensor:
            # we need to transpose as self attention always treats
            # dim -2 as the sequence dimension
            if self.multiquery_item_attention_for_test_set:
                if single_eval_pos < x.shape[1]:
                    new_x_test = self.self_attn_between_items(
                        x[:, single_eval_pos:].transpose(1, 2),
                        x[:, :single_eval_pos].transpose(1, 2)
                        if single_eval_pos
                        else None,
                        save_peak_mem_factor=save_peak_mem_factor,
                        cache_kv=False,
                        add_input=True,
                        allow_inplace=True,
                        use_cached_kv=not single_eval_pos,
                        reuse_first_head_kv=True,
                        use_second_set_of_queries=self.two_sets_of_queries,
                    ).transpose(1, 2)
                else:
                    new_x_test = None

                if single_eval_pos:
                    new_x_train = self.self_attn_between_items(
                        x[:, :single_eval_pos].transpose(1, 2),
                        x[:, :single_eval_pos].transpose(1, 2),
                        save_peak_mem_factor=save_peak_mem_factor,
                        cache_kv=cache_trainset_representation,
                        only_cache_first_head_kv=True,
                        add_input=True,
                        allow_inplace=True,
                        use_cached_kv=False,
                    ).transpose(1, 2)
                else:
                    new_x_train = None

                return torch.cat(
                    [x_ for x_ in [new_x_train, new_x_test] if x_ is not None],
                    dim=1,
                )

            attention_src_x = None
            if att_src is not None:
                attention_src_x = att_src.transpose(1, 2)
            elif single_eval_pos:
                attention_src_x = x[:, :single_eval_pos].transpose(1, 2)

            return self.self_attn_between_items(
                x.transpose(1, 2),
                attention_src_x,
                save_peak_mem_factor=save_peak_mem_factor,
                cache_kv=cache_trainset_representation and single_eval_pos,
                add_input=True,
                allow_inplace=True,
                use_cached_kv=cache_trainset_representation and not single_eval_pos,
            ).transpose(1, 2)

        mlp_save_peak_mem_factor = (
            save_peak_mem_factor * 8 if save_peak_mem_factor is not None else None
        )

        sublayers = []
        if self.self_attn_between_features is not None:
            sublayers.append(attn_between_features)
        else:
            assert state.shape[2] == 1, (
                "If there is no attention between features, the number of feature"
                " blocks must be 1."
            )

        sublayers += [
            attn_between_items,
            partial(
                self.mlp.__call__,
                save_peak_mem_factor=mlp_save_peak_mem_factor
                if (
                    mlp_save_peak_mem_factor is not None
                    and state.numel() // state.shape[-1] // mlp_save_peak_mem_factor
                )
                > 32
                else None,
                add_input=True,
                allow_inplace=True,
            ),
        ]

        if self.second_mlp is not None:
            sublayers.insert(
                1,
                partial(
                    self.second_mlp.__call__,
                    save_peak_mem_factor=mlp_save_peak_mem_factor,
                    add_input=True,
                    allow_inplace=True,
                ),
            )

        for sublayer, layer_norm in zip(sublayers, self.layer_norms):
            if self.pre_norm:
                raise AssertionError(
                    "Pre-norm implementation is wrong, as the residual should never"
                    " be layer normed here.",
                )
                state = layer_norm(
                    state,
                    allow_inplace=True,
                    save_peak_mem_factor=save_peak_mem_factor,
                )

            state = sublayer(state)
            if not self.pre_norm:
                state = layer_norm(
                    state,
                    allow_inplace=True,
                    save_peak_mem_factor=save_peak_mem_factor,
                )

        return state

    def empty_trainset_representation_cache(self) -> None:
        """Empty the trainset representation cache."""
        self.self_attn_between_items.empty_kv_cache()

        # TODO: This could be None but this just ignored that fact here.
        assert self.self_attn_between_features is not None
        self.self_attn_between_features.empty_kv_cache()  # not necessary, just in case


DEFAULT_EMSIZE = 128


@contextmanager
def isolate_torch_rng(seed: int, device: torch.device) -> Generator[None, None, None]:
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)


class LayerStack(nn.Module):
    """Similar to nn.Sequential, but with support for passing keyword arguments
    to layers and stacks the same layer multiple times.
    """

    def __init__(
        self,
        *,
        layer_creator: Callable[[], nn.Module],
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        half_layers: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if half_layers:
            assert self.min_num_layers_layer_dropout == self.num_layers, (
                "half_layers only works without layer dropout"
            )
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                low=self.min_num_layers_layer_dropout,
                high=self.num_layers + 1,
                size=(1,),
            ).item()

        for layer in self.layers[:n_layers]:
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)  # type: ignore
            else:
                x = layer(x, **kwargs)

        return x


class PerFeatureTransformer(nn.Module):
    """A Transformer model processes a token per feature and sample.

    This model extends the standard Transformer architecture to operate on a
    per-feature basis.
    It allows for processing each feature separately while still leveraging the
    power of self-attention.

    The model consists of an encoder, decoder, and optional components such
    as a feature positional embedding and a separate decoder for each feature.
    """

    # TODO: Feel like this could be simplified a lot from this part downwards
    def __init__(  # noqa: C901, D417, PLR0913
        self,
        *,
        encoder: nn.Module | None = None,
        ninp: int = DEFAULT_EMSIZE,
        nhead: int = 4,
        nhid: int = DEFAULT_EMSIZE * 4,
        nlayers: int = 10,
        y_encoder: nn.Module | None = None,
        decoder_dict: dict[str, tuple[type[nn.Module] | None, int]] | None = None,
        init_method: str | None = None,
        activation: Literal["gelu", "relu"] = "gelu",
        recompute_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
        repeat_same_layer: bool = False,
        dag_pos_enc_dim: int = 0,
        features_per_group: int = 1,
        feature_positional_embedding: (
            Literal[
                "normal_rand_vec",
                "uni_rand_vec",
                "learned",
                "subspace",
            ]
            | None
        ) = None,
        zero_init: bool = True,
        use_separate_decoder: bool = False,
        nlayers_decoder: int | None = None,
        use_encoder_compression_layer: bool = False,
        precomputed_kv: (
            list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
        cache_trainset_representation: bool = False,
        seed: int | None = None,
        # TODO: List explicitly
        **layer_kwargs: Any,
    ):
        """Initializes the PerFeatureTransformer module.

        Args:
            encoder:
                Pass a nn.Module that takes in a batch of sequences of inputs and
                returns something of the shape (seq_len, batch_size, ninp)
            ninp: Input dimension, also called the embedding dimension
            nhead: Number of attention heads
            nhid: Hidden dimension in the MLP layers
            nlayers:
                Number of layers, each consisting of a multi-head attention layer and
                an MLP layer
            y_encoder:
                A nn.Module that takes in a batch of sequences of outputs and
                returns something of the shape (seq_len, batch_size, ninp)
            decoder_dict: Document this (TODO)
            activation: An activation function, "gelu" or "relu"
            recompute_layer:
                If True, the transformer layers will be recomputed on each
                forward pass in training. This is useful to save memory.
            min_num_layers_layer_dropout:
                If this is set, it enables to drop the last
                layers randomly during training up to this number.
            repeat_same_layer:
                If True, the same layer will be used for all layers.
                This is useful to save memory on weights.
            features_per_group:
                If > 1, the features will be grouped into groups of this
                size and the attention is across groups.
            feature_positional_embedding:
                There is a risk that our models confuse
                features with each other. This positional embedding is added to the
                features to help the model distinguish them.
                We recommend setting this to "subspace".
            zero_init:
                If True, the last sublayer of each attention and MLP layer will
                be initialized with zeros.
                Thus, the layers will start out as identity functions.
            seed: The seed to use for the random number generator.
            use_separate_decoder: If True, the decoder will be separate from the encoder
            nlayers_decoder:
                If use_separate_decoder is True, this is the number of
                layers in the decoder. The default is to use 1/3 of the layers for the
                decoder and 2/3 for the encoder.
            use_encoder_compression_layer: Experimental
            precomputed_kv: Experimental
            layer_kwargs:
                TODO: document.
                for now have a look at layer.py:PerFeatureEncoderLayer.
        """
        if decoder_dict is None:
            decoder_dict = {"standard": (None, 1)}

        super().__init__()

        if encoder is None:
            encoder = SequentialEncoder(
                LinearInputEncoderStep(
                    num_features=1,
                    emsize=DEFAULT_EMSIZE,
                    replace_nan_by_zero=False,
                    bias=True,
                    in_keys=("main",),
                    out_keys=("output",),
                ),
            )

        if y_encoder is None:
            y_encoder = SequentialEncoder(
                NanHandlingEncoderStep(),
                LinearInputEncoderStep(
                    num_features=2,
                    emsize=DEFAULT_EMSIZE,
                    replace_nan_by_zero=False,
                    bias=True,
                    out_keys=("output",),
                    in_keys=("main", "nan_indicators"),
                ),
            )

        self.encoder = encoder
        self.y_encoder = y_encoder
        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.init_method = init_method
        self.features_per_group = features_per_group
        self.cache_trainset_representation = cache_trainset_representation
        self.cached_embeddings: torch.Tensor | None = None

        layer_creator = lambda: PerFeatureEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            activation=activation,
            zero_init=zero_init,
            precomputed_kv=(
                precomputed_kv.pop(0) if precomputed_kv is not None else None
            ),
            **layer_kwargs,
        )
        if repeat_same_layer:
            layer = layer_creator()
            layer_creator = lambda: layer

        nlayers_encoder = nlayers
        if use_separate_decoder and nlayers_decoder is None:
            nlayers_decoder = max((nlayers // 3) * 1, 1)
            nlayers_encoder = max((nlayers // 3) * 2, 1)

        self.transformer_encoder = LayerStack(
            layer_creator=layer_creator,
            num_layers=nlayers_encoder,
            recompute_each_layer=recompute_layer,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
        )

        self.transformer_decoder = None
        if use_separate_decoder:
            assert nlayers_decoder is not None
            self.transformer_decoder = LayerStack(
                layer_creator=layer_creator,
                num_layers=nlayers_decoder,
            )

        self.global_att_embeddings_for_compression = None
        if use_encoder_compression_layer:
            assert use_separate_decoder
            num_global_att_tokens_for_compression = 512

            self.global_att_embeddings_for_compression = nn.Embedding(
                num_global_att_tokens_for_compression,
                ninp,
            )

            self.encoder_compression_layer = LayerStack(
                layer_creator=layer_creator,
                num_layers=2,
            )

        initialized_decoder_dict = {}
        for decoder_key in decoder_dict:
            decoder_model, decoder_n_out = decoder_dict[decoder_key]
            if decoder_model is None:
                initialized_decoder_dict[decoder_key] = nn.Sequential(
                    nn.Linear(ninp, nhid),
                    nn.GELU(),
                    nn.Linear(nhid, decoder_n_out),
                )
            else:
                initialized_decoder_dict[decoder_key] = decoder_model(
                    ninp,
                    nhid,
                    decoder_n_out,
                )
        self.decoder_dict = nn.ModuleDict(initialized_decoder_dict)

        self.feature_positional_embedding = feature_positional_embedding
        if feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)

        self.dag_pos_enc_dim = dag_pos_enc_dim
        self.cached_feature_positional_embeddings: torch.Tensor | None = None
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)  # noqa: S311

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        """Sets the save_peak_mem_factor for all layers.

        This factor controls how much memory is saved during the forward pass
        in inference mode.

        Setting this factor > 1 will cause the model to save more memory during
        the forward pass in inference mode.

        A value of 8 is good for a 4x larger width in the fully-connected layers.
        and yields a situation were we need around
        `2*num_features*num_items*emsize*2` bytes of memory

        for a forward pass (using mixed precision).

        WARNING: It should only be used with post-norm.

        Args:
            factor: The save_peak_mem_factor to set. Recommended value is 8.
        """
        for layer in self.transformer_encoder.layers:
            assert hasattr(
                layer,
                "save_peak_mem_factor",
            ), "Layer does not have save_peak_mem_factor"
            layer.save_peak_mem_factor = factor  # type: ignore

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("features_per_group", 1)
        state.setdefault("feature_positional_embedding", None)
        super().__setstate__(state)

    # TODO(eddiebergman): Can we just replace this with specific calls
    # such as forward, forward_with_test, forward_with_style?
    # The documentation generator complains about this function because we are
    # documenting parameters that don't exist in the signature
    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:  # noqa: D417
        """Performs a forward pass through the model.

        This method supports multiple calling conventions:

        - `model((x,y), **kwargs)`
        - `model(train_x, train_y, test_x, **kwargs)`
        - `model((style,x,y), **kwargs)`

        Args:
            train_x: torch.Tensor | None
                The input data for the training set.
            train_y: torch.Tensor | None
                The target data for the training set.
            test_x: torch.Tensor | None
                The input data for the test set.
            x: torch.Tensor
                The input data.
            y: torch.Tensor | None
                The target data.
            style: torch.Tensor | None
                The style vector.
            single_eval_pos: int
                The position to evaluate at.
            only_return_standard_out: bool
                Whether to only return the standard output.
            data_dags: Any
                The data DAGs for each example.
            categorical_inds: list[int]
                The indices of categorical features.
            freeze_kv: bool
                Whether to freeze the key and value weights.

        Returns:
            The output of the model, which can be a tensor or a dictionary of tensors.
        """
        half_layers = kwargs.pop("half_layers", False)
        assert half_layers is False

        supported_kwargs = {
            "only_return_standard_out",
            "style",
            "data_dags",
            "categorical_inds",
            "freeze_kv",
            "train_x",
            "train_y",
            "test_x",
            "single_eval_pos",
        }
        spurious_kwargs = set(kwargs.keys()) - supported_kwargs
        assert not spurious_kwargs, spurious_kwargs

        if args == () and all(k in kwargs for k in ("train_x", "train_y", "test_x")):
            assert "single_eval_pos" not in kwargs
            x = kwargs.pop("train_x")
            train_y = kwargs.pop("train_y")
            test_x = kwargs.pop("test_x")
            if test_x is not None:
                x = torch.cat((x, test_x), dim=0)
            return self._forward(x, train_y, single_eval_pos=len(train_y), **kwargs)

        if len(args) == 2:
            x, y = args
            return self._forward(x, y, **kwargs)

        if len(args) == 3:
            style, x, y = args
            return self._forward(x, y, style=style, **kwargs)

        raise ValueError("Unrecognized input. Please follow the doc string.")

    def _forward(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor | dict,
        # TODO(eddiebergman): Not sure if it can be None but the function seems to
        # indicate it could
        y: torch.Tensor | dict | None,
        *,
        single_eval_pos: int | None = None,
        only_return_standard_out: bool = True,
        style: torch.Tensor | None = None,
        data_dags: list[Any] | None = None,
        categorical_inds: list[int] | None = None,
        half_layers: bool = False,
    ) -> Any | dict[str, torch.Tensor]:
        """The core forward pass of the model.

        Args:
            x: The input data. Shape: `(seq_len, batch_size, num_features)`
            y: The target data. Shape: `(seq_len, batch_size)`
            single_eval_pos:
                The position to evaluate at. If `None`, evaluate at all positions.
            only_return_standard_out: Whether to only return the standard output.
            style: The style vector.
            data_dags: The data DAGs for each example in the batch.
            categorical_inds: The indices of categorical features.
            half_layers: Whether to use half the layers.

        Returns:
            A dictionary of output tensors.
        """
        assert style is None
        if self.cache_trainset_representation:
            if not single_eval_pos:  # none or 0
                assert y is None
        else:
            assert y is not None
            assert single_eval_pos

        single_eval_pos_ = single_eval_pos or 0
        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
        else:
            x = {"main": x}
        seq_len, batch_size, num_features = x["main"].shape

        if y is None:
            # TODO: check dtype.
            y = torch.zeros(
                0,
                batch_size,
                device=x["main"].device,
                dtype=x["main"].dtype,
            )

        if isinstance(y, dict):
            assert "main" in set(y.keys()), f"Main must be in input keys: {y.keys()}."
        else:
            y = {"main": y}

        for k in x:
            num_features_ = x[k].shape[2]

            # pad to multiple of features_per_group
            missing_to_next = (
                self.features_per_group - (num_features_ % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            seq_len,
                            batch_size,
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,
                )

        # Splits up the input into subgroups
        for k in x:
            x[k] = einops.rearrange(
                x[k],
                "s b (f n) -> b s f n",
                n=self.features_per_group,
            )  # s b f -> b s #groups #features_per_group

        # We have to re-work categoricals based on the subgroup they fall into.
        categorical_inds_to_use: list[list[int]] | None = None
        if categorical_inds is not None:
            new_categorical_inds = []
            n_subgroups = x["main"].shape[2]

            for subgroup in range(n_subgroups):
                subgroup_lower = subgroup * self.features_per_group
                subgroup_upper = (subgroup + 1) * self.features_per_group
                subgroup_indices = [
                    i - subgroup_lower
                    for i in categorical_inds
                    if subgroup_lower <= i < subgroup_upper
                ]
                new_categorical_inds.append(subgroup_indices)

            categorical_inds_to_use = new_categorical_inds

        for k in y:
            if y[k].ndim == 1:
                y[k] = y[k].unsqueeze(-1)
            if y[k].ndim == 2:
                y[k] = y[k].unsqueeze(-1)  # s b -> s b 1

            y[k] = y[k].transpose(0, 1)  # s b 1 -> b s 1

            if y[k].shape[1] < x["main"].shape[1]:
                assert (
                    y[k].shape[1] == single_eval_pos_
                    or y[k].shape[1] == x["main"].shape[1]
                )
                assert k != "main" or y[k].shape[1] == single_eval_pos_, (
                    "For main y, y must not be given for target"
                    " time steps (Otherwise the solution is leaked)."
                )
                if y[k].shape[1] == single_eval_pos_:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],
                                x["main"].shape[1] - y[k].shape[1],
                                y[k].shape[2],
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,
                    )

            y[k] = y[k].transpose(0, 1)  # b s 1 -> s b 1

        # making sure no label leakage ever happens
        y["main"][single_eval_pos_:] = torch.nan

        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del y
        if torch.isnan(embedded_y).any():
            raise ValueError(
                f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers"
                " to the ys that are not fully provided (test set missing)",
            )

        extra_encoders_args = {}
        if categorical_inds_to_use is not None and isinstance(
            self.encoder,
            SequentialEncoder,
        ):
            extra_encoders_args["categorical_inds"] = categorical_inds_to_use

        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")

        embedded_x = einops.rearrange(
            self.encoder(
                x,
                single_eval_pos=single_eval_pos_,
                cache_trainset_representation=self.cache_trainset_representation,
                **extra_encoders_args,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )  # b s f 1 -> b s f e
        del x

        embedded_x, embedded_y = self.add_embeddings(
            embedded_x,
            embedded_y,
            data_dags=data_dags,
            num_features=num_features,
            seq_len=seq_len,
            cache_embeddings=(
                self.cache_trainset_representation and single_eval_pos is not None
            ),
            use_cached_embeddings=(
                self.cache_trainset_representation and single_eval_pos is None
            ),
        )
        del data_dags

        # b s f e + b s 1 e -> b s f+1 e
        embedded_input = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)

        if torch.isnan(embedded_input).any():
            raise ValueError(
                f"There should be no NaNs in the encoded x and y."
                "Check that you do not feed NaNs or use a NaN-handling enocder."
                "Your embedded x and y returned the following:"
                f"{torch.isnan(embedded_x).any()=} | {torch.isnan(embedded_y).any()=}",
            )
        del embedded_y, embedded_x

        encoder_out = self.transformer_encoder(
            (
                embedded_input
                if not self.transformer_decoder
                else embedded_input[:, :single_eval_pos_]
            ),
            single_eval_pos=single_eval_pos,
            half_layers=half_layers,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # b s f+1 e -> b s f+1 e

        # If we are using a decoder
        if self.transformer_decoder:
            assert not half_layers
            assert encoder_out.shape[1] == single_eval_pos_

            if self.global_att_embeddings_for_compression is not None:
                # TODO: fixed number of compression tokens
                train_encoder_out = self.encoder_compression_layer(
                    self.global_att_embeddings_for_compression,
                    att_src=encoder_out[:, single_eval_pos_],
                    single_eval_pos=single_eval_pos_,
                )

            test_encoder_out = self.transformer_decoder(
                embedded_input[:, single_eval_pos_:],
                single_eval_pos=0,
                att_src=encoder_out,
            )
            encoder_out = torch.cat([encoder_out, test_encoder_out], 1)
            del test_encoder_out

        del embedded_input

        # out: s b e
        test_encoder_out = encoder_out[:, single_eval_pos_:, -1].transpose(0, 1)

        if only_return_standard_out:
            assert self.decoder_dict is not None
            output_decoded = self.decoder_dict["standard"](test_encoder_out)
        else:
            output_decoded = (
                {k: v(test_encoder_out) for k, v in self.decoder_dict.items()}
                if self.decoder_dict is not None
                else {}
            )

            # out: s b e
            train_encoder_out = encoder_out[:, :single_eval_pos_, -1].transpose(0, 1)
            output_decoded["train_embeddings"] = train_encoder_out
            output_decoded["test_embeddings"] = test_encoder_out

        return output_decoded

    def add_embeddings(  # noqa: C901, PLR0912
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        data_dags: Iterable | None,
        num_features: int,
        seq_len: int,
        cache_embeddings: bool = False,
        use_cached_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_cached_embeddings and self.cached_embeddings is not None:
            assert data_dags is None, (
                "Caching embeddings is not supported with data_dags at this point."
            )
            x += self.cached_embeddings[None, None]
            return x, y

        with isolate_torch_rng(self.seed, device=x.device):
            if self.feature_positional_embedding == "normal_rand_vec":
                embs = torch.randn(
                    (x.shape[2], x.shape[3]),
                    device=x.device,
                    dtype=x.dtype,
                )
                x += embs[None, None]
            elif self.feature_positional_embedding == "uni_rand_vec":
                embs = (
                    torch.rand(
                        (x.shape[2], x.shape[3]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    * 2
                    - 1
                )
                x += embs[None, None]
            elif self.feature_positional_embedding == "learned":
                w = self.feature_positional_embedding_embeddings.weight
                embs = w[
                    torch.randint(
                        0,
                        w.shape[0],
                        (x.shape[2],),
                    )
                ]
                x += embs[None, None]
            elif self.feature_positional_embedding == "subspace":
                embs = torch.randn(
                    (x.shape[2], x.shape[3] // 4),
                    device=x.device,
                    dtype=x.dtype,
                )
                embs = self.feature_positional_embedding_embeddings(embs)
                x += embs[None, None]
            elif self.feature_positional_embedding is None:
                embs = None
            else:
                raise ValueError(f"Unknown {self.feature_positional_embedding=}")

        self.cached_embeddings = None
        if cache_embeddings and embs is not None:
            assert data_dags is None, (
                "Caching embeddings is not supported with data_dags at this point."
            )
            self.cached_embeddings = embs

        # TODO(old) should this go into encoder?
        # could also be made a bit more concise by moving down to operate on full_x
        if data_dags is not None:
            for b_i, data_dag in enumerate(data_dags):
                # TODO(eddibergman): Very inneficient way to make a full connect
                # DiGraph
                g_ = data_dag.copy()
                while _networkx_add_direct_connections(g_):
                    pass

                subgraph = g_.subgraph(  # type: ignore
                    [
                        n
                        for n, info in g_.nodes.items()
                        if (info["is_feature"] or info["is_target"])
                    ],
                )
                k = self.dag_pos_enc_dim
                assert k > 0
                _add_pos_emb(subgraph, k=k)

                graph_pos_embs_features = torch.zeros((num_features, k))
                graph_pos_embs_targets = torch.zeros((1, k))  # shape: (num_targets, k)

                for node_info in subgraph.nodes.values():
                    for feature_idx in node_info.get("feature_idxs", []):
                        graph_pos_embs_features[feature_idx] = node_info[
                            "positional_encoding"
                        ]
                    for target_idx in node_info.get("target_idxs", []):
                        graph_pos_embs_targets[target_idx] = node_info[
                            "positional_encoding"
                        ]

                graph_pos_embs_targets -= graph_pos_embs_features.mean(0, keepdim=True)
                graph_pos_embs_features -= graph_pos_embs_features.mean(0, keepdim=True)

                graph_pos_embs_features = graph_pos_embs_features[None].expand(
                    seq_len,
                    -1,
                    -1,
                )
                x[b_i, :, :, :k] += graph_pos_embs_features.to(y.device, y.dtype)

                graph_pos_embs_targets = (
                    graph_pos_embs_targets[None].expand(seq_len, -1, -1).squeeze(-2)
                )
                y[b_i, :, :k] += graph_pos_embs_targets.to(y.device, y.dtype)
        else:
            assert not hasattr(self, "dag_pos_enc_dim") or not self.dag_pos_enc_dim

        return x, y

    def empty_trainset_representation_cache(self) -> None:
        for layer in (self.transformer_decoder or self.transformer_encoder).layers:
            layer.empty_trainset_representation_cache()


def _networkx_add_direct_connections(graph) -> bool:
    added_connection = False
    # Get the list of nodes
    nodes = list(graph.nodes)

    # Iterate over each node
    for node in nodes:
        # Get the direct neighbors of the current node
        neighbors = list(graph.neighbors(node))

        # Iterate over the neighbors of the current node
        for neighbor in neighbors:
            # Get the neighbors of the neighbor
            second_neighbors = list(graph.neighbors(neighbor))

            # Iterate over the neighbors of the neighbor
            for second_neighbor in second_neighbors:
                # Add a direct edge from the current node to the second neighbor,
                # if it doesn't exist already
                if second_neighbor not in graph.neighbors(node):
                    graph.add_edge(node, second_neighbor)

                    added_connection = True
    return added_connection


def _add_pos_emb(
    graph,
    *,
    is_undirected: bool = False,
    k: int = 20,
) -> None:
    from scipy.sparse.linalg import eigs, eigsh

    eig_fn = eigs if not is_undirected else eigsh

    # L = nx.directed_laplacian_matrix(graph)
    L = 0.0
    np.nan_to_num(L, nan=0.0, copy=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eig_vals, eig_vecs = eig_fn(  # type: ignore
            L,
            k=k + 1,
            which="SR" if not is_undirected else "SA",
            return_eigenvectors=True,
        )

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe_ = torch.from_numpy(eig_vecs[:, 1 : k + 1])
        pe = torch.zeros(len(eig_vecs), k)
        pe[:, : pe_.shape[1]] = pe_
        sign = -1 + 2 * torch.randint(0, 2, (k,))
        pe *= sign

        # TODO(old) Double check the ordering is right
        for n, pe_ in zip(graph.nodes(), pe):
            graph.nodes[n]["positional_encoding"] = pe_
