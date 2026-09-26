"""fp8 WMMA expert path for GGUF MoE layers on gfx1201 (kernels: sglang-gfx1201/moe_fp8/moe_fp8.cu).

gate_up and down run as W8A8 fp8 grouped GEMMs: the GGUF expert weights are unpacked to fp8 in
registers inside the GEMM (one scale per layer tensor, set at load), activations are quantized
per row (tokens for gate_up, pairs for down, silu(gate) * up fused with the quantization).
Measured on Qwen3.6-35B-A3B layers (gfx1201): whole layer 5.36 -> ~2.8 ms at 4096 tokens,
1.66 -> ~1.05 ms at 128. Types: Q4_K, Q5_K, padded Q6_K (raw Q6_K is repacked at load).
"""
import torch

from sglang.srt.layers.quantization import moe_fp8 as _ext
from sglang.srt.layers.quantization.gguf_fp8 import _SB_BYTES, quant_act, row_amax

SUPPORTED = {12, 13, 1014}


def weight_scale(w: torch.Tensor, qtype: int) -> float:
    """One fp8 scale for all experts of a [E, N, bytes] GGUF tensor (margin keeps |w/s| < 448)."""
    E, N, B = w.shape
    K = B // _SB_BYTES[qtype] * 256
    return float(row_amax(w.view(E * N, B), qtype, K).max()) / 448.0 * 1.001


def _configs(pairs: int, inter: int):
    """(BM, WN) for gate_up and (BM, WN, N-tiles per workgroup) for down, from the gfx1201 sweep
    (tests/moe_fp8/gemm_sweep.py): big blocks once experts see ~64+ pairs each."""
    big = pairs >= 16384
    g = (64, 32) if big else (64, 16)
    if inter != 512:                       # the N-loop kernel is built for K = 512 only
        return g, (32, 32, 0)
    return g, ((64, 16, 16) if big else (32, 32, 2))


def fused_moe_fp8(x, w13, w2, topk_weights, topk_ids, q13, q2, ws13, ws2, moe_align_block_size):
    T, H = x.shape
    E, N13, _ = w13.shape
    top_k = topk_ids.shape[1]
    P = T * top_k
    inter = N13 // 2
    (bm1, wn1), (bm2, wn2, nt2) = _configs(P, inter)
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    xq, sx = quant_act(x, True)
    sid, eid, ntpp = moe_align_block_size(ids, bm1, E)
    h = torch.empty((P, N13), dtype=torch.bfloat16, device=x.device)
    _ext.moe_fp8_gemm(xq.view(torch.uint8), sx.view(-1), w13, sid, eid, ntpp, P, q13, N13, H, top_k,
                      ws13, 1.0 / ws13, None, bm1, wn1, h, 0)
    aq, sa = _ext.silu_mul_quant_fp8(h)
    if bm2 != bm1:
        sid, eid, ntpp = moe_align_block_size(ids, bm2, E)
    y = torch.empty((P, H), dtype=torch.bfloat16, device=x.device)
    _ext.moe_fp8_gemm(aq, sa, w2, sid, eid, ntpp, P, q2, H, inter, 1, ws2, 1.0 / ws2,
                      topk_weights.reshape(-1).to(torch.float32).contiguous(), bm2, wn2, y, nt2)
    return y.view(T, top_k, H).sum(dim=1).to(x.dtype)
