"""Fused dequantize-in-tile GEMM for GGUF k-quant weights (Triton).

y[M, N] = x[M, K] @ dequant(W)[N, K]^T  for W in Q5_K / Q6_K / Q8_0 (ggml layouts).

Same idea as sglang's awq_gemm_triton: every program owns BLOCK_N output rows and
a K-range, streams the packed bytes ONCE, unpacks them into a bf16 [BLOCK_N, 128]
tile and feeds tl.dot (bf16 WMMA on RDNA4). Weight traffic is therefore
independent of M, unlike the ported mmvq whose cost is linear in M.

Layouts (ggml-common.h), K in super-blocks of 256 unless noted:
  Q5_K (176 B): d f16 | dmin f16 | scales[12] (6-bit sc/min packed) | qh[32] | qs[128]
      elem k: j=k//32, i=k%32; q = ((qs[(j//2)*32+i] >> 4*(j%2)) & 0xF) | (((qh[i] >> j) & 1) << 4)
      w = d*sc[j]*q - dmin*m[j]
  Q6_K (210 B): ql[128] | qh[64] | scales[16] i8 | d f16
      elem k: ql[(j//4)*64 + (j%2)*32 + i] >> 4*((j//2)%2), qh[(j//4)*32 + i] >> 2*(j%4)
      q = (lo | hi<<4) - 32 ; w = d * scales[k//16] * q
  Q8_0 (34 B per 32): d f16 | qs[32] i8 ; w = d * q

Reference for every formula: gguf-py quants.py (dequantize_blocks), which the test
harness compares against.
"""
import torch
import triton
import triton.language as tl

Q8_0, Q4_K, Q5_K, Q6_K, Q6_K_PAD = 8, 12, 13, 14, 1014
# per 256 weights; raw Q6_K (210 B, 2-byte aligned) is ~8x slower than the padded 224 B layout
_BYTES_PER_SB = {Q4_K: 144, Q5_K: 176, Q6_K: 210, Q6_K_PAD: 224, Q8_0: 34 * 8}
FUSED_TYPES = set(_BYTES_PER_SB)


@triton.jit
def _f16_from_bytes(lo, hi):
    # two uint8 -> float16 value (bitcast)
    v = (hi.to(tl.int32) << 8) | lo.to(tl.int32)
    return v.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _unpack_q5_k(base, j, ii):
    """base: [BLOCK_N,1] uint8 ptrs to the super-block; j: sub-block 0..7 (constexpr); ii: [1,32]."""
    qs = tl.load(base + 48 + (j // 2) * 32 + ii).to(tl.int32)
    qh = tl.load(base + 16 + ii).to(tl.int32)
    q = ((qs >> (4 * (j % 2))) & 0xF) | (((qh >> j) & 1) << 4)
    if j < 4:
        sc = tl.load(base + 4 + j).to(tl.int32) & 0x3F
        mn = tl.load(base + 8 + j).to(tl.int32) & 0x3F
    else:
        md = tl.load(base + 12 + (j - 4)).to(tl.int32)
        sc = (md & 0x0F) | ((tl.load(base + 4 + (j - 4)).to(tl.int32) >> 2) & 0x30)
        mn = (md >> 4) | ((tl.load(base + 8 + (j - 4)).to(tl.int32) >> 2) & 0x30)
    d = _f16_from_bytes(tl.load(base + 0), tl.load(base + 1))
    dmin = _f16_from_bytes(tl.load(base + 2), tl.load(base + 3))
    return (d * sc.to(tl.float32)) * q.to(tl.float32) - dmin * mn.to(tl.float32)


@triton.jit
def _unpack_q4_k(base, j, ii):
    """Q4_K (144 B): d f16 | dmin f16 | scales[12] | qs[128]; like Q5_K without the high-bit plane."""
    qs = tl.load(base + 16 + (j // 2) * 32 + ii).to(tl.int32)
    q = (qs >> (4 * (j % 2))) & 0xF
    if j < 4:
        sc = tl.load(base + 4 + j).to(tl.int32) & 0x3F
        mn = tl.load(base + 8 + j).to(tl.int32) & 0x3F
    else:
        md = tl.load(base + 12 + (j - 4)).to(tl.int32)
        sc = (md & 0x0F) | ((tl.load(base + 4 + (j - 4)).to(tl.int32) >> 2) & 0x30)
        mn = (md >> 4) | ((tl.load(base + 8 + (j - 4)).to(tl.int32) >> 2) & 0x30)
    d = _f16_from_bytes(tl.load(base + 0), tl.load(base + 1))
    dmin = _f16_from_bytes(tl.load(base + 2), tl.load(base + 3))
    return (d * sc.to(tl.float32)) * q.to(tl.float32) - dmin * mn.to(tl.float32)


@triton.jit
def _unpack_q6_k(base, j, ii):
    ql = tl.load(base + (j // 4) * 64 + (j % 2) * 32 + ii).to(tl.int32)
    qh = tl.load(base + 128 + (j // 4) * 32 + ii).to(tl.int32)
    q = (((ql >> (4 * ((j // 2) % 2))) & 0xF) | (((qh >> (2 * (j % 4))) & 0x3) << 4)) - 32
    s0 = tl.load(base + 192 + j * 2).to(tl.int8).to(tl.int32)             # two int8 scales per sub-block
    s1 = tl.load(base + 193 + j * 2).to(tl.int8).to(tl.int32)
    sc = tl.where(ii < 16, s0, s1)
    d = _f16_from_bytes(tl.load(base + 208), tl.load(base + 209))
    return d * sc.to(tl.float32) * q.to(tl.float32)


@triton.jit
def _unpack_q8_0(base, j, ii):
    q = tl.load(base + j * 34 + 2 + ii).to(tl.int8).to(tl.int32)
    d = _f16_from_bytes(tl.load(base + j * 34), tl.load(base + j * 34 + 1))
    return d * q.to(tl.float32)


@triton.jit
def _gguf_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_om,
    QTYPE: tl.constexpr, SB_BYTES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_mask = rn < N
    m_mask = rm < M
    num_sb = K // 256
    sb_per_split = (num_sb + SPLIT_K - 1) // SPLIT_K
    sb_start = pid_k * sb_per_split
    sb_end = tl.minimum(num_sb, sb_start + sb_per_split)
    rn_safe = tl.where(n_mask, rn, 0)
    ii = tl.arange(0, 32)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for sb in range(sb_start, sb_end):
        base = w_ptr + rn_safe[:, None] * stride_wn + sb * SB_BYTES   # [BLOCK_N,1]
        for j in tl.static_range(8):
            if QTYPE == 12:
                w = _unpack_q4_k(base, j, ii)
            elif QTYPE == 13:
                w = _unpack_q5_k(base, j, ii)
            elif QTYPE == 14 or QTYPE == 1014:
                w = _unpack_q6_k(base, j, ii)
            else:
                w = _unpack_q8_0(base, j, ii)
            xt = tl.load(
                x_ptr + rm[:, None] * stride_xm + sb * 256 + j * 32 + ii,
                mask=m_mask[:, None], other=0.0,
            )
            acc += tl.dot(xt.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
    out = out_ptr + pid_k * (M * stride_om) + rm[:, None] * stride_om + rn[None, :]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def _pick_grid(N: int, num_sb: int):
    """Measured on gfx1201: bn=64 is the efficient tile (bn 16/32 are 2.5-3x slower);
    split_k=1 saves the partial-sum + cast launches where the grid covers the CUs twice
    or K is short. Otherwise below 128 programs split K: at M=4, N=5120 (80 programs)
    ffn_down K=17408 320 -> 234 us (in a HIP graph), o_proj K=6144 117 -> 102 us, ssm_out
    Q8_0 100 -> 91 us; N=4096 K=5120 85 -> 79 us; but N=4096 K=2048 41 -> 49 us."""
    progs = triton.cdiv(N, 64)
    if progs >= 128 or (progs >= 64 and num_sb < 16):
        return 64, 1
    for split_k in (2, 4, 8, 16):
        if split_k > num_sb:
            break
        if progs * split_k >= 256:
            return 64, split_k
    return (64 if N >= 64 else 16), max(1, min(16, num_sb))


def gguf_gemm_triton(x: torch.Tensor, qweight: torch.Tensor, qtype: int, N: int, K: int,
                     block_m: int = 16, block_n: int | None = None, split_k: int | None = None,
                     num_warps: int = 4, num_stages: int = 1) -> torch.Tensor:
    """x: [M, K] (bf16/fp16), qweight: [N, bytes] uint8 contiguous. Returns [M, N] in x.dtype."""
    assert qtype in FUSED_TYPES and K % 256 == 0
    M = x.shape[0]
    x = x.contiguous()
    bn, sk = _pick_grid(N, K // 256)
    block_n = block_n or bn
    split_k = split_k or sk
    if split_k == 1:
        y = torch.empty((M, N), dtype=x.dtype, device=x.device)   # direct bf16 store, no reduction
        out = y
    else:
        out = torch.empty((split_k, M, N), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, block_n), split_k, triton.cdiv(M, block_m))
    _gguf_gemm_kernel[grid](
        x, qweight, out, M, N, K,
        x.stride(0), qweight.stride(0), N,
        QTYPE=qtype, SB_BYTES=_BYTES_PER_SB[qtype],
        BLOCK_M=block_m, BLOCK_N=block_n, SPLIT_K=split_k, num_warps=num_warps, num_stages=num_stages,
    )
    if split_k == 1:
        return y
    return out.sum(0).to(x.dtype)
