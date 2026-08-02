"""GPU token masking implemented as a fused Triton kernel."""

from __future__ import annotations

from numbers import Integral

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # Keep the module importable in CPU-only environments.
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: ImportError | None = exc
else:
    _TRITON_IMPORT_ERROR = None


if triton is not None:

    @triton.jit
    def _token_masking_kernel(
        input_ids_ptr,
        mask_ptr,
        output_ptr,
        mask_token_id,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Replace selected input elements with ``mask_token_id``."""
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        in_bounds = offsets < n_elements

        input_ids = tl.load(input_ids_ptr + offsets, mask=in_bounds)
        should_mask = tl.load(mask_ptr + offsets, mask=in_bounds, other=False)
        output = tl.where(should_mask, mask_token_id, input_ids)
        tl.store(output_ptr + offsets, output, mask=in_bounds)

else:
    _token_masking_kernel = None


_SUPPORTED_TOKEN_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _validate_inputs(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
) -> None:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if input_ids.dtype not in _SUPPORTED_TOKEN_DTYPES:
        raise TypeError(f"input_ids must have an integer dtype, got {input_ids.dtype}")
    if mask.dtype is not torch.bool:
        raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}")
    if input_ids.shape != mask.shape:
        raise ValueError(
            "input_ids and mask must have the same shape, got "
            f"{tuple(input_ids.shape)} and {tuple(mask.shape)}"
        )
    if input_ids.device != mask.device:
        raise ValueError(
            "input_ids and mask must be on the same device, got "
            f"{input_ids.device} and {mask.device}"
        )
    if input_ids.device.type != "cuda":
        raise ValueError("the Triton token masking kernel requires CUDA tensors")
    if not isinstance(mask_token_id, Integral):
        raise TypeError("mask_token_id must be an integer")

    dtype_limits = torch.iinfo(input_ids.dtype)
    if not dtype_limits.min <= mask_token_id <= dtype_limits.max:
        raise ValueError(
            f"mask_token_id {mask_token_id} cannot be represented by {input_ids.dtype}"
        )


def mask_tokens(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Return ``input_ids`` with positions selected by ``mask`` replaced.

    The operation is out-of-place and preserves the input shape and dtype. Inputs
    may be non-contiguous; they are made contiguous before the one-dimensional
    kernel launch. ``mask`` must be a boolean CUDA tensor with the same shape and
    device as ``input_ids``.

    Args:
        input_ids: Integer CUDA tensor containing token IDs or labels.
        mask: Boolean CUDA tensor. ``True`` positions are replaced.
        mask_token_id: Integer replacement value, such as a tokenizer's mask ID
            or ``-100`` for ignored labels.

    Returns:
        A new contiguous tensor containing the masked values.
    """
    _validate_inputs(input_ids, mask, mask_token_id)

    if _token_masking_kernel is None:
        raise RuntimeError(
            "Triton is required for mask_tokens; install it in the CUDA environment"
        ) from _TRITON_IMPORT_ERROR

    contiguous_input = input_ids.contiguous()
    contiguous_mask = mask.contiguous()
    output = torch.empty_like(contiguous_input)
    n_elements = contiguous_input.numel()
    if n_elements == 0:
        return output

    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    with torch.cuda.device(input_ids.device):
        _token_masking_kernel[grid](
            contiguous_input,
            contiguous_mask,
            output,
            int(mask_token_id),
            n_elements,
            BLOCK_SIZE=block_size,
        )
    return output


def token_masking(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Backward-compatible name for :func:`mask_tokens`."""
    return mask_tokens(input_ids, mask, mask_token_id)


__all__ = ["mask_tokens", "token_masking"]
