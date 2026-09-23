# Triton-side readers for the fp4_mx_block16 KV pool: packed E2M1 nibbles
# ([tokens, heads, head_dim // 2] uint8, element 2j in the low nibble of byte j)
# with one E8M0 scale per 16 values ([tokens, heads * head_dim // 16] uint8).
# Layout contract: FP4MXBlock16KVQuantizeUtil.batched_quantize (kvfp4_tensor.py).
import triton
import triton.language as tl


@triton.jit
def fp4_nibble_to_bf16(nib, scale_u8):
    """E2M1 nibble (uint8 in [0, 16)) times 2**(scale - 127), built as bf16 bits.
    Integer-only: bf16 shares fp32's exponent width, so the E8M0 scale adds into
    the exponent field. (int16 arithmetic here trips an LLVM truncate assert in
    the AMD backend of Triton 3.7; keep int32 and narrow at the end.)"""
    sign = (nib & 8).to(tl.int32) << 12
    e = ((nib >> 1) & 3).to(tl.int32)
    m = (nib & 1).to(tl.int32)
    s = scale_u8.to(tl.int32)
    # normal: (1 + m/2) * 2**(e - 1 + s - 127); subnormal: m * 0.5 * 2**(s - 127)
    normal = ((e - 1 + s) << 7) | (m << 6)
    sub = tl.where(m == 1, tl.maximum(s - 1, 0) << 7, 0)
    bits = tl.where(e == 0, sub, normal) | sign
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def fp4_even(packed, scale):
    return fp4_nibble_to_bf16(packed & 0xF, scale)


@triton.jit
def fp4_odd(packed, scale):
    return fp4_nibble_to_bf16((packed >> 4) & 0xF, scale)


@triton.jit
def _expand_scale_rows(sc, BLOCK_N: tl.constexpr, N_BLOCKS: tl.constexpr):
    """[N_BLOCKS, BLOCK_N] scale tile -> [N_BLOCKS * 8, BLOCK_N], each row repeated
    8 times (one scale block of 16 values = 8 packed bytes)."""
    sc3 = tl.broadcast_to(tl.reshape(sc, [N_BLOCKS, 1, BLOCK_N]), [N_BLOCKS, 8, BLOCK_N])
    return tl.reshape(sc3, [N_BLOCKS * 8, BLOCK_N])


@triton.jit
def _expand_scale_cols(sc, BLOCK_N: tl.constexpr, N_BLOCKS: tl.constexpr):
    """[BLOCK_N, N_BLOCKS] -> [BLOCK_N, N_BLOCKS * 8], each column repeated 8 times."""
    sc3 = tl.broadcast_to(tl.reshape(sc, [BLOCK_N, N_BLOCKS, 1]), [BLOCK_N, N_BLOCKS, 8])
    return tl.reshape(sc3, [BLOCK_N, N_BLOCKS * 8])


@triton.jit
def load_k_fp4_packed(
    K_Buffer,
    K_Scale,
    kv_loc,  # [BLOCK_N] token rows
    mask_n,  # [BLOCK_N]
    cur_kv_head,
    stride_buf_kbs,
    stride_buf_kh,
    stride_ks_bs,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Transposed packed K tile [HEAD_DIM // 2, BLOCK_N] uint8 and its scale tile
    expanded to the same shape. Decode one half at a time with fp4_even / fp4_odd
    (k[2j, n] = even[j, n], k[2j + 1, n] = odd[j, n]) so both bf16 halves are
    never live together."""
    HALF: tl.constexpr = HEAD_DIM // 2
    N_BLOCKS: tl.constexpr = HEAD_DIM // 16
    offs_j = tl.arange(0, HALF)
    offs_b = tl.arange(0, N_BLOCKS)
    offs_pk = kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_j[:, None]
    packed = tl.load(K_Buffer + offs_pk, mask=mask_n[None, :], other=0)
    # byte j holds elements 2j and 2j + 1, both in scale block j // 8
    offs_s = kv_loc[None, :] * stride_ks_bs + cur_kv_head * N_BLOCKS + offs_b[:, None]
    scale = _expand_scale_rows(
        tl.load(K_Scale + offs_s, mask=mask_n[None, :], other=127), BLOCK_N, N_BLOCKS
    )
    return packed, scale


@triton.jit
def load_v_fp4(
    V_Buffer,
    V_Scale,
    kv_loc,  # [BLOCK_N] token rows
    mask_n,  # [BLOCK_N]
    cur_kv_head,
    stride_buf_vbs,
    stride_buf_vh,
    stride_vs_bs,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """V tile [BLOCK_N, V_HEAD_DIM] bf16 in natural element order."""
    HALF: tl.constexpr = V_HEAD_DIM // 2
    N_BLOCKS: tl.constexpr = V_HEAD_DIM // 16
    offs_j = tl.arange(0, HALF)
    offs_b = tl.arange(0, N_BLOCKS)
    offs_pk = kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_j[None, :]
    packed = tl.load(V_Buffer + offs_pk, mask=mask_n[:, None], other=0)
    offs_s = kv_loc[:, None] * stride_vs_bs + cur_kv_head * N_BLOCKS + offs_b[None, :]
    scale = _expand_scale_cols(
        tl.load(V_Scale + offs_s, mask=mask_n[:, None], other=127), BLOCK_N, N_BLOCKS
    )
    even = fp4_nibble_to_bf16(packed & 0xF, scale)
    odd = fp4_nibble_to_bf16((packed >> 4) & 0xF, scale)
    # join puts (even, odd) in a trailing dim of 2; the reshape interleaves them
    return tl.reshape(tl.join(even, odd), [BLOCK_N, V_HEAD_DIM])


@triton.jit
def fp4_k_interleaved(packed, scale, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr):
    """Full transposed K tile [HEAD_DIM, BLOCK_N] bf16 from load_k_fp4_packed's
    outputs, natural element order (for kernels that cannot afford Q halves)."""
    even = fp4_even(packed, scale)
    odd = fp4_odd(packed, scale)
    return tl.reshape(tl.permute(tl.join(even, odd), (0, 2, 1)), [HEAD_DIM, BLOCK_N])


@triton.jit
def load_q_fp4_pair(
    Q,
    offs_row,  # [M] row offsets (already multiplied by the row stride)
    mask_row,  # [M]
    HEAD_DIM: tl.constexpr,
):
    """Q columns split by parity to pair with load_k_fp4_pair: two [M, HEAD_DIM // 2]."""
    HALF: tl.constexpr = HEAD_DIM // 2
    offs_j = tl.arange(0, HALF)
    q_even = tl.load(Q + offs_row[:, None] + (2 * offs_j)[None, :], mask=mask_row[:, None], other=0.0)
    q_odd = tl.load(Q + offs_row[:, None] + (2 * offs_j + 1)[None, :], mask=mask_row[:, None], other=0.0)
    return q_even.to(tl.bfloat16), q_odd.to(tl.bfloat16)


# ---------------------------------------------------------------------------
# Quantize: one Triton launch in place of the torch.compile'd
# FP4MXBlock16KVQuantizeUtil.batched_quantize (same rounding: E8M0 scale =
# ceil(log2(blockmax / 6)), magnitude by the E2M1 midpoint thresholds).
# ---------------------------------------------------------------------------
@triton.jit
def _quantize_fp4_mx16_kernel(
    X, Packed, Scales, stride_xb, stride_xm, stride_pb, stride_pm, stride_sb,
    N: tl.constexpr,
):
    """One program per (batch row, head): N values -> N // 2 packed bytes and
    N // 16 scales. X is [B, M, N] with unit stride on N."""
    b = tl.program_id(0)
    m = tl.program_id(1)
    HALF: tl.constexpr = N // 2
    NB: tl.constexpr = N // 16
    offs = tl.arange(0, N)
    x = tl.load(X + b * stride_xb + m * stride_xm + offs).to(tl.float32)
    blocks = tl.reshape(x, [NB, 16])
    bmax = tl.max(tl.abs(blocks), axis=1)  # [NB]
    scale_exp = tl.ceil(tl.log2(tl.maximum(bmax / 6.0, 1e-10)))
    scaled = tl.reshape(blocks / tl.exp2(scale_exp)[:, None], [N])
    a = tl.abs(scaled)
    mag = (
        (a >= 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a >= 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32) + (a >= 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
        + (a >= 5.0).to(tl.int32)
    )
    nib = tl.where(scaled < 0, 8, 0) + mag  # [N] in [0, 16)
    pairs = tl.reshape(nib, [HALF, 2])
    lo, hi = tl.split(pairs)
    packed = (lo | (hi << 4)).to(tl.uint8)
    tl.store(Packed + b * stride_pb + m * stride_pm + tl.arange(0, HALF), packed)
    tl.store(Scales + b * stride_sb + m * NB + tl.arange(0, NB), (scale_exp + 127).to(tl.uint8))


def quantize_fp4_mx16(x):
    """[B, M, N] (bf16/fp16/fp32) -> (packed uint8 [B, M, N // 2], scales uint8 [B, M * N // 16])."""
    import torch

    B, M, N = x.shape
    assert N % 16 == 0 and N == triton.next_power_of_2(N), N
    packed = torch.empty((B, M, N // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((B, M * N // 16), dtype=torch.uint8, device=x.device)
    if B == 0:
        return packed, scales
    _quantize_fp4_mx16_kernel[(B, M)](
        x, packed, scales, x.stride(0), x.stride(1), packed.stride(0), packed.stride(1), scales.stride(0),
        N=N, num_warps=1,
    )
    return packed, scales
