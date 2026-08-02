# Triton kernel for token-level masking
# This kernel applies a binary mask to a 2D attention score matrix.
# `scores` shape: [seq_len, seq_len]
# `mask` shape: [seq_len]  (1 for keep, 0 for mask out)
# The output writes back into `scores` in-place, setting masked rows/cols to -inf.

import triton
import triton.language as tl

@triton.jit
def token_mask_kernel(
    scores_ptr,  # *float32
    mask_ptr,    # *int32
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    # Guard against out-of-bounds
    mask = tl.load(mask_ptr + row, mask=row < seq_len, other=0).to(tl.float32)
    # Load a block of the scores matrix
    scores = tl.load(scores_ptr + row[:, None] * seq_len + col[None, :],
                     mask=(row[:, None] < seq_len) & (col[None, :] < seq_len),
                     other=0.0)
    # Apply mask: if mask[row] == 0, set entire row to -inf; similarly for columns
    row_mask = mask[:, None]
    col_mask = mask[None, :]
    combined_mask = row_mask * col_mask
    # Set masked positions to -1e9 (approx -inf for fp32)
    scores = tl.where(combined_mask > 0, scores, tl.full(scores.shape, -1e9, dtype=tl.float32))
    # Write back
    tl.store(scores_ptr + row[:, None] * seq_len + col[None, :], scores,
             mask=(row[:, None] < seq_len) & (col[None, :] < seq_len))

def apply_token_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply token-level mask to attention scores using Triton.

    Args:
        scores (torch.Tensor): [seq_len, seq_len] attention score matrix.
        mask (torch.Tensor): [seq_len] binary mask (1 = keep, 0 = mask out).
    Returns:
        torch.Tensor: masked scores tensor (modified in-place).
    """
    import torch
    assert scores.dim() == 2 and mask.dim() == 1
    seq_len = scores.size(0)
    grid = lambda META: ( (seq_len + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'], )
    token_mask_kernel[grid](scores_ptr=scores, mask_ptr=mask, seq_len=seq_len)
    return scores
