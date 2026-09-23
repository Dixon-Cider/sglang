"""Fused grouped GEMM for GGUF-quantized MoE experts (Triton).

For every (token, expert) pair chosen by top-k routing, computes
    h[pair]   = x[token] @ dequant(W13[expert]).T          (gate|up, N = 2*inter)
    y[pair]   = act(h[pair]) @ dequant(W2[expert]).T       (down,   N = hidden)
with the same dequantize-in-tile scheme as gguf_triton.py: a program owns one
BLOCK_M block of pairs that share an expert (from moe_align_block_size) and a
BLOCK_N tile of that expert's rows; the packed weight bytes are read once per
block and dotted against all pairs in it (bf16 WMMA). Replaces the ported
ggml_moe_a8_vec, which reads the expert weights once per pair per column.
"""
import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.gguf_triton import _unpack_q4_k, _unpack_q5_k, _unpack_q6_k, _unpack_q8_0, _BYTES_PER_SB, FUSED_TYPES


@triton.jit
def _gguf_moe_gemm_kernel(
    x_ptr, w_ptr, out_ptr, tw_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    num_valid_pairs, N, K,
    stride_xm, stride_we, stride_wn, stride_om,
    top_k: tl.constexpr, X_IS_PAIR: tl.constexpr,
    QTYPE: tl.constexpr, SB_BYTES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return
    expert = tl.load(expert_ids_ptr + pid_m)
    if expert == -1:
        return
    offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_token_ids_ptr + offs)                 # flattened index into topk_ids
    m_mask = pair < num_valid_pairs
    pair_safe = tl.where(m_mask, pair, 0)
    xrow = pair_safe if X_IS_PAIR else pair_safe // top_k        # stage 2 reads per-pair rows
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N
    rn_safe = tl.where(n_mask, rn, 0)
    ii = tl.arange(0, 32)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    wbase = w_ptr + expert.to(tl.int64) * stride_we + rn_safe[:, None] * stride_wn
    for sb in range(0, K // 256):
        base = wbase + sb * SB_BYTES
        for j in tl.static_range(8):
            if QTYPE == 12:
                w = _unpack_q4_k(base, j, ii)
            elif QTYPE == 13:
                w = _unpack_q5_k(base, j, ii)
            elif QTYPE == 14 or QTYPE == 1014:
                w = _unpack_q6_k(base, j, ii)
            else:
                w = _unpack_q8_0(base, j, ii)
            xt = tl.load(x_ptr + xrow[:, None] * stride_xm + sb * 256 + j * 32 + ii, mask=m_mask[:, None], other=0.0)
            acc += tl.dot(xt.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
    if X_IS_PAIR:   # down projection: fold the routing weight of each pair into its row
        acc *= tl.load(tw_ptr + pair_safe, mask=m_mask, other=0.0)[:, None]
    out = out_ptr + pair_safe[:, None] * stride_om + rn[None, :]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def _launch(x, w, out, tw, sorted_token_ids, expert_ids, num_tokens_post_padded, num_valid_pairs, top_k, x_is_pair, qtype, N, K, block_m, block_n, num_warps):
    grid = (triton.cdiv(int(sorted_token_ids.numel()), block_m), triton.cdiv(N, block_n))
    _gguf_moe_gemm_kernel[grid](
        x, w, out, tw, sorted_token_ids, expert_ids, num_tokens_post_padded,
        num_valid_pairs, N, K,
        x.stride(0), w.stride(0), w.stride(1), out.stride(0),
        top_k=top_k, X_IS_PAIR=x_is_pair, QTYPE=qtype, SB_BYTES=_BYTES_PER_SB[qtype],
        BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps,
    )


# 64 x 64 tiles, 8 warps (gfx1201, Qwen3.6-A3B experts, 4096-token chunk = 128 pairs/expert):
# 5.5 ms/layer vs 52 at 16 x 64 / 4 warps -- the dequantized tile is shared by 4x the pairs and
# the unpack spreads over 8 warps; 2.5 vs 13 ms at 32 pairs/expert. 128-row tiles exceed LDS.
def fused_moe_gguf_triton(x, w13, w2, topk_weights, topk_ids, qtype13, qtype2, moe_align_block_size, act,
                          block_m: int = 64, block_n: int = 64, num_warps: int = 8):
    """x [T, hidden] bf16; w13 [E, 2*inter, bytes]; w2 [E, hidden, bytes] (uint8, contiguous).
    Returns [T, hidden] in x.dtype. `act` maps [P, 2*inter] -> [P, inter] (silu_and_mul)."""
    T, hidden = x.shape
    E, N13, _ = w13.shape
    top_k = topk_ids.shape[1]
    P = T * top_k
    _, ts13 = (256, _BYTES_PER_SB[qtype13])
    K13 = w13.shape[2] // ts13 * 256
    assert K13 == hidden
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_m, E)
    h = torch.empty((P, N13), dtype=x.dtype, device=x.device)
    _launch(x.contiguous(), w13, h, None, sorted_token_ids, expert_ids, num_tokens_post_padded, P, top_k, False, qtype13, N13, K13, block_m, block_n, num_warps)
    a = act(h)                                                   # [P, inter]
    inter = a.shape[1]
    K2 = w2.shape[2] // _BYTES_PER_SB[qtype2] * 256
    assert K2 == inter, (K2, inter)
    y = torch.empty((P, hidden), dtype=x.dtype, device=x.device)
    tw = topk_weights.reshape(-1).to(torch.float32).contiguous()
    _launch(a.contiguous(), w2, y, tw, sorted_token_ids, expert_ids, num_tokens_post_padded, P, top_k, True, qtype2, hidden, K2, block_m, block_n, num_warps)
    return y.view(T, top_k, hidden).sum(dim=1)
