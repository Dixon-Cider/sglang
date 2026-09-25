"""FP8 (e4m3fn) prefill GEMM for GGUF weights on gfx1201.

y[M, N] = x[M, K] @ dequant(W)[N, K]^T, computed as
    wq = fp8(dequant(W) / s_w)            s_w: one fp32 scale per layer (all shards), set at load
    xq = fp8(x / s_x)                     s_x: per token [M, 1] (default) or per tensor
    y  = _scaled_mm(xq, wq^T) * s_x * s_w (hipBLASLt tensorwise fp8, then one row multiply)

Why this shape (measured on gfx1201, kernel-opt/logs/fp8_probe0.txt): hipBLASLt's rowwise
fp8 epilogue runs ~160 TFLOPS, tensorwise ~225, bf16 ~118; tensorwise + one broadcast
multiply lands at 205-220. A per-layer weight scale costs almost nothing in accuracy because
e4m3 is floating point (4.5 decades of normal range); the per-token activation scale is the
one that matters (massive-activation tokens), so it is kept and applied after the GEMM.

The dequantize writes fp8 straight from the packed GGUF bytes (half the bytes of the old
bf16 temp), all shards of a merged layer into one buffer, so one GEMM and no torch.cat.
"""
import torch
import triton
import triton.language as tl

try:
    from sglang.srt.layers.quantization.gguf_triton import (
        _BYTES_PER_SB, _f16_from_bytes, _unpack_q4_k, _unpack_q5_k, _unpack_q6_k, _unpack_q8_0,
    )
except ImportError:  # standalone use from sglang-gfx1201/gguf_triton/
    from gguf_triton import (
        _BYTES_PER_SB, _f16_from_bytes, _unpack_q4_k, _unpack_q5_k, _unpack_q6_k, _unpack_q8_0,
    )

IQ4_XS = 23
_SB_BYTES = dict(_BYTES_PER_SB)
_SB_BYTES[IQ4_XS] = 136
FP8_TYPES = set(_SB_BYTES)
FP8_MAX = 448.0
F8 = torch.float8_e4m3fn


@triton.jit
def _iq4nl_level(q):
    """ggml kvalues_iq4nl[q], q in [0, 16): four int8 levels per int32 word, no table in memory."""
    hi = q >> 2
    w = tl.where(hi == 0, -1079142271, tl.where(hi == 1, -152379953, tl.where(hi == 2, 639175937, 1901675829)))
    return ((w >> ((q & 3) * 8)) << 24) >> 24


@triton.jit
def _unpack_iq4_xs(base, j, i16):
    """IQ4_XS (136 B): d f16 | scales_h u16 | scales_l[4] | qs[128]. Sub-block j as two
    [BLOCK_N, 16] halves: elements 0..15 (low nibbles) and 16..31 (high nibbles)."""
    qb = tl.load(base + 8 + j * 16 + i16).to(tl.int32)
    lsl = (tl.load(base + 4 + j // 2).to(tl.int32) >> (4 * (j % 2))) & 0xF
    sh = tl.load(base + 2).to(tl.int32) | (tl.load(base + 3).to(tl.int32) << 8)
    ls = lsl | (((sh >> (2 * j)) & 3) << 4)
    dl = _f16_from_bytes(tl.load(base + 0), tl.load(base + 1)) * (ls - 32).to(tl.float32)
    return dl * _iq4nl_level(qb & 0xF).to(tl.float32), dl * _iq4nl_level(qb >> 4).to(tl.float32)


@triton.jit
def _to_fp8(w, inv):
    return tl.clamp(w * inv, -448.0, 448.0).to(tl.float8e4nv)


@triton.jit
def _fp8x4(w, inv):
    """[BN, W] f32 -> [BN, W // 4] int32, four e4m3 bytes per dword (little endian).
    Triton 3.7 lowers f32 -> e4m3 to ~10 compare/select ops per value on gfx12 (3.4x slower
    than this kernel's byte floor); v_cvt_pk_fp8_f32 does two values per instruction (RNE,
    bit-exact vs torch). A pack=4 uint8 asm output does not register-allocate, so the tile is
    split into its four byte lanes and the asm takes four f32 operands."""
    BN: tl.constexpr = w.shape[0]
    W: tl.constexpr = w.shape[1]
    v = tl.clamp(w * inv, -448.0, 448.0)
    a, b = tl.split(tl.reshape(v, (BN, W // 4, 2, 2)))
    e0, e2 = tl.split(a)
    e1, e3 = tl.split(b)
    return tl.inline_asm_elementwise(
        "v_cvt_pk_fp8_f32 $0, $1, $2\n\tv_cvt_pk_fp8_f32 $0, $3, $4 op_sel:[0,0,1]",
        "=&v,v,v,v,v", [e0, e1, e2, e3], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _dequant_fp8_kernel(w_ptr, out_ptr, inv_ptr, N, stride_wn, stride_on,
                        QTYPE: tl.constexpr, SB_BYTES: tl.constexpr, BLOCK_N: tl.constexpr):
    """grid (cdiv(N, BLOCK_N), K // 256): one 256-wide super-block of BLOCK_N rows per program.
    out_ptr is the fp8 buffer viewed as int32 (stride_on in dwords)."""
    pid_n = tl.program_id(0)
    sb = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N
    rn_safe = tl.where(n_mask, rn, 0)
    inv = tl.load(inv_ptr)
    base = w_ptr + rn_safe[:, None].to(tl.int64) * stride_wn + sb * SB_BYTES
    orow = out_ptr + rn[:, None].to(tl.int64) * stride_on + sb * 64
    ii = tl.arange(0, 32)[None, :]
    i16 = tl.arange(0, 16)[None, :]
    d4 = tl.arange(0, 4)[None, :]
    d8 = tl.arange(0, 8)[None, :]
    for j in tl.static_range(8):
        if QTYPE == 23:
            lo, hi = _unpack_iq4_xs(base, j, i16)
            tl.store(orow + j * 8 + d4, _fp8x4(lo, inv), mask=n_mask[:, None])
            tl.store(orow + j * 8 + 4 + d4, _fp8x4(hi, inv), mask=n_mask[:, None])
        else:
            if QTYPE == 12:
                w = _unpack_q4_k(base, j, ii)
            elif QTYPE == 13:
                w = _unpack_q5_k(base, j, ii)
            elif QTYPE == 14 or QTYPE == 1014:
                w = _unpack_q6_k(base, j, ii)
            else:
                w = _unpack_q8_0(base, j, ii)
            tl.store(orow + j * 8 + d8, _fp8x4(w, inv), mask=n_mask[:, None])


@triton.jit
def _row_amax_kernel(w_ptr, amax_ptr, N, K, stride_wn,
                     QTYPE: tl.constexpr, SB_BYTES: tl.constexpr, BLOCK_N: tl.constexpr):
    """max |dequant(W)| per row; load time only."""
    pid_n = tl.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N
    rn_safe = tl.where(n_mask, rn, 0)
    ii = tl.arange(0, 32)[None, :]
    i16 = tl.arange(0, 16)[None, :]
    m = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for sb in range(0, K // 256):
        base = w_ptr + rn_safe[:, None].to(tl.int64) * stride_wn + sb * SB_BYTES
        for j in tl.static_range(8):
            if QTYPE == 23:
                lo, hi = _unpack_iq4_xs(base, j, i16)
                m = tl.maximum(m, tl.maximum(tl.max(tl.abs(lo), 1), tl.max(tl.abs(hi), 1)))
            else:
                if QTYPE == 12:
                    w = _unpack_q4_k(base, j, ii)
                elif QTYPE == 13:
                    w = _unpack_q5_k(base, j, ii)
                elif QTYPE == 14 or QTYPE == 1014:
                    w = _unpack_q6_k(base, j, ii)
                else:
                    w = _unpack_q8_0(base, j, ii)
                m = tl.maximum(m, tl.max(tl.abs(w), 1))
    tl.store(amax_ptr + rn, m, mask=n_mask)


@triton.jit
def _act_amax_kernel(x_ptr, amax_ptr, K, stride_xm, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    m = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row.to(tl.int64) * stride_xm + offs, mask=offs < K, other=0.0).to(tl.float32)
        m = tl.maximum(m, tl.abs(v))
    tl.store(amax_ptr + row, tl.max(m, 0))


@triton.jit
def _act_quant_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm, s_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    inv = 1.0 / tl.load(s_ptr + row * s_stride)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row.to(tl.int64) * stride_xm + offs, mask=offs < K, other=0.0).to(tl.float32)
        tl.store(q_ptr + row.to(tl.int64) * K + offs, _to_fp8(v, inv), mask=offs < K)


def row_amax(qweight: torch.Tensor, qtype: int, K: int) -> torch.Tensor:
    """[N] fp32 max |w| per row of a GGUF weight [N, bytes] (uint8)."""
    N = qweight.shape[0]
    out = torch.empty(N, dtype=torch.float32, device=qweight.device)
    _row_amax_kernel[(triton.cdiv(N, 64),)](qweight, out, N, K, qweight.stride(0),
                                            QTYPE=qtype, SB_BYTES=_SB_BYTES[qtype], BLOCK_N=64, num_warps=4)
    return out


def dequant_fp8(qweight: torch.Tensor, qtype: int, K: int, inv_scale: torch.Tensor,
                out: torch.Tensor | None = None) -> torch.Tensor:
    """fp8(dequant(W) * inv_scale) -> [N, K] float8_e4m3fn. inv_scale: 0-d/1-elem fp32 device tensor."""
    N = qweight.shape[0]
    if out is None:
        out = torch.empty((N, K), dtype=F8, device=qweight.device)
    assert K % 256 == 0 and out.stride(1) == 1 and out.stride(0) % 4 == 0
    # 16 rows x 8 warps: the unpack is bound by ~50 dependent byte loads per program, so more,
    # smaller programs win (gfx1201, ffn_up Q5_K 17408x5120: 64x4 1.17 ms, 16x8 0.50 ms)
    _dequant_fp8_kernel[(triton.cdiv(N, 16), K // 256)](
        qweight, out.view(torch.int32), inv_scale, N, qweight.stride(0), out.stride(0) // 4,
        QTYPE=qtype, SB_BYTES=_SB_BYTES[qtype], BLOCK_N=16, num_warps=8)
    return out


def quant_act(x: torch.Tensor, per_token: bool = True):
    """x [M, K] (row stride arbitrary, unit column stride) -> (xq [M, K] fp8, scale [M,1] or 0-d fp32)."""
    M, K = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    amax = torch.empty(M, dtype=torch.float32, device=x.device)
    _act_amax_kernel[(M,)](x, amax, K, x.stride(0), BLOCK=1024, num_warps=4)
    if per_token:
        s = amax.clamp_min_(1e-6).div_(FP8_MAX).view(M, 1)
        s_stride = 1
    else:
        s = (amax.amax().clamp_min(1e-6) / FP8_MAX)
        s_stride = 0
    q = torch.empty((M, K), dtype=F8, device=x.device)
    _act_quant_kernel[(M,)](x, q, s, K, x.stride(0), s_stride, BLOCK=1024, num_warps=4)
    return q, s


_ONE = {}
_W_CHUNK_BYTES = 512 << 20


def gguf_fp8_linear(x: torch.Tensor, shards, w_scale: torch.Tensor, w_inv: torch.Tensor,
                    K: int, per_token: bool = True) -> torch.Tensor:
    """shards: list of (qweight [rows, bytes] uint8, qtype); w_scale / w_inv: 1-elem fp32 device
    tensors shared by all shards. Returns x @ W^T as [M, sum(rows)] in x.dtype."""
    M = x.shape[0]
    N = sum(w.shape[0] for w, _ in shards)
    xq, sx = quant_act(x, per_token)
    one = _ONE.get(x.device)
    if one is None:
        one = _ONE[x.device] = torch.ones((), dtype=torch.float32, device=x.device)
    sa = one if per_token else sx
    sb = w_scale.reshape(())
    if N * K <= _W_CHUNK_BYTES:
        wq = torch.empty((N, K), dtype=F8, device=x.device)
        r = 0
        for w, qt in shards:
            dequant_fp8(w, qt, K, w_inv, wq[r:r + w.shape[0]])
            r += w.shape[0]
        y = torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=sb, out_dtype=x.dtype)
    else:  # very large layers: row chunks (per shard, split further if needed)
        y = torch.empty((M, N), dtype=x.dtype, device=x.device)
        rows_per = max(256, _W_CHUNK_BYTES // K // 256 * 256)   # _scaled_mm needs N % 16 == 0
        r = 0
        for w, qt in shards:
            for r0 in range(0, w.shape[0], rows_per):
                wc = dequant_fp8(w[r0:r0 + rows_per], qt, K, w_inv)
                n = wc.shape[0]
                y[:, r + r0:r + r0 + n] = torch._scaled_mm(xq, wc.t(), scale_a=sa, scale_b=sb, out_dtype=x.dtype)
            r += w.shape[0]
    if per_token:
        y.mul_(sx)
    return y
