"""Split-KV (flash-decode) attention for EAGLE speculative *verify*.

Only valid when speculative ``topk == 1`` (the EAGLE tree reduces to a pure
causal chain); the caller gates on that. ``topk > 1`` trees fall back to
``extend_attention_fwd``.

On the Triton backend, EAGLE target-verify runs through the prefill
``extend_attention_fwd``, which loops the (long) prefix KV serially per
(sequence, head). With only a few draft-token queries, that leaves the GPU
memory system far under-utilized at long context. This kernel instead splits
the prefix KV across parallel programs (flash-decode style) and combines the
partials with a log-sum-exp merge, then handles the small causal draft-draft
block -- recovering memory bandwidth on the verify path.

Two Triton kernels:
  * ``_verify_prefix_stage1``: split-KV over the shared prefix. Applies the fp8
    dequant multipliers ``k_scale`` (on the QK score) and ``v_scale`` (on the
    prefix output), matching ``extend_attention_fwd``'s ``_fwd_kernel``
    (qk *= sm_scale * k_scale; acc += dot(p, v) * v_scale on the prefix loop;
    NO scaling on the draft-draft loop, whose K/V are the fresh bf16 draft
    tensors, not the fp8 pool). fp8 K/V buffers are handled by casting q to the
    buffer dtype before the dot (mirrors ``q.to(k.dtype)`` in the baseline).
  * ``_verify_combine_stage2``: combines the prefix splits (LSE merge) with the
    small causal draft-draft block and writes the output.

``verify_splitkv_fwd(...)`` takes the SAME positional args as
``extend_attention_fwd``; it runs the split-KV path when it can serve the case
bit-equivalently and returns True, otherwise returns False (doing nothing) so
the caller falls back to ``extend_attention_fwd``. Supported case: causal
(topk=1) verify with a constant per-sequence extend length, no sinks /
sliding-window / logit-cap / xai-temperature. Correctness is never violated.
"""

import os

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.kv_fp4 import (
    fp4_even,
    fp4_odd,
    load_k_fp4_packed,
    load_q_fp4_pair,
    load_v_fp4,
)

from sglang.srt.utils import is_hip

_MIN_BLOCK_KV = 32

# AMD/CDNA-only Triton launch hints (waves_per_eu, matrix_instr_nonkdim); NVIDIA's
# Triton rejects these kwargs, so only pass them on ROCm. In production this kernel
# is dispatched only on AMD (see TritonAttnBackend); keeping it NV-safe lets the
# numerics test run on the CUDA CI lane.
_IS_HIP = is_hip()
# gfx1201 fp4 verify stage 1 (sglang-gfx1201/fp4-attn-gfx1201.sh builds fp4_attn.so)
_fp4_attn_hip = None
if _IS_HIP:
    try:
        from sglang.kernels.ops.attention import fp4_attn as _fp4_attn_hip
    except ImportError:
        _fp4_attn_hip = None
_AMD_LAUNCH_KWARGS = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16} if _IS_HIP else {}

# Block-size config keyed on head_dim. The (BLOCK_N, num_warps) tile that best
# hides latency depends on head_dim: at head_dim=256 (Qwen3 family) a narrower
# BLOCK_N with more warps wins, since the 256-wide QK/PV tiles are register
# heavy. head_dim=256 is the value validated on MI350X; other head dims use a
# conservative default. Block size affects PERFORMANCE only, never correctness
# (any valid block size produces the same result).
DEFAULT_N_SPLITS = 8
DEFAULT_BLOCK_N = 32
DEFAULT_NUM_WARPS = 4
_BLOCK_CONFIG = {
    # head_dim: (BLOCK_N, num_warps)
    256: (32, 4),
}


def block_config(head_dim):
    """Return (BLOCK_N, num_warps) for a head_dim; default for untuned dims."""
    return _BLOCK_CONFIG.get(head_dim, (DEFAULT_BLOCK_N, DEFAULT_NUM_WARPS))


# ---------------------------------------------------------------------------
# Adaptive N_SPLITS.
# ---------------------------------------------------------------------------
# The prefix split-KV stage launches a (bs, h_q, N_SPLITS) grid; each (b,h,s)
# program handles kv_len_per_split = cdiv(cdiv(seqlen, N_SPLITS), MIN)*MIN keys.
# A fixed N_SPLITS=16 over-splits short/mid contexts (each split does too little
# work -> launch + reduction overhead dominates) and under-splits very long ones
# (too few parallel waves to saturate the device, raising tail latency on the
# slow split). Mirror the decode kernel's intent (decode_attention.py
# get_num_kv_splits): pick the split count per-dispatch from the representative
# sequence length, growing gradually with seqlen and capped at MAX.
#
# CRITICAL: this must be computed from STATIC shapes only (no .item()/.cpu()
# sync), because the verify/draft-extend step runs inside a captured HIP graph
# where a device->host copy raises hipErrorStreamCaptureUnsupported. We use the
# average prefix length = kv_indices.shape[0] / bs, which is a pure python int
# from tensor shapes -- no device read. N_SPLITS is then a power of two so the
# stage2 reduction tile (tl.arange(0, N_SPLITS)) stays cheap.
#
# Split-count bounds (internal constants). MAX=16 is the MI350X cap: 32
# oversubscribes the device and regresses, per tuning.
ADAPTIVE_SPLITS = True
MAX_N_SPLITS = 16
MIN_N_SPLITS = 4


def choose_n_splits(avg_seqlen):
    """Pick N_SPLITS (power of two, in [MIN_N_SPLITS, MAX_N_SPLITS]) from the
    average prefix length. Tuned by the real-shape sweep (head_dim=256, BS*H_Q
    =128 base programs on ~132 CUs):

        ctx  <  4k -> 4   (short: extra splits add launch/reduction overhead)
        4k <= ctx < 8k -> 8   (sweet spot: best across 1k-16k in the sweep)
        ctx >= 8k      -> 16  (long: a few more splits help latency-bound tail)

    Never 32 (4096 grid blocks oversubscribes the device and regresses, per the
    sweep). Computed from a static shape (avg prefix = kv_indices.shape[0]/bs),
    so it is HIP-graph-capture safe (no device->host sync)."""
    if not ADAPTIVE_SPLITS:
        return DEFAULT_N_SPLITS
    s = int(avg_seqlen)
    if s < 4096:
        n = 4
    elif s < 8192:
        n = 8
    elif s < 32768:
        n = 16
    elif s < 65536:
        n = 32  # reached only when the cap allows it
    else:
        n = 64
    if n < MIN_N_SPLITS:
        n = MIN_N_SPLITS
    if n > MAX_N_SPLITS:
        n = MAX_N_SPLITS
    return n


@triton.jit
def _verify_prefix_stage1(
    Q,  # [extend_tokens, H_Q, D]
    K_Buffer,  # [pool_tokens, H_KV, D]
    V_Buffer,  # [pool_tokens, H_KV, Dv]
    sm_scale,
    k_scale,  # fp8 dequant multiplier for prefix K (1.0 if bf16)
    v_scale,  # fp8 dequant multiplier for prefix V (1.0 if bf16)
    qo_indptr,  # [BS+1] int32  -> rows of Q (draft queries)
    kv_indptr,  # [BS+1] int32  -> rows of kv_indices (prefix)
    kv_indices,  # [sum prefix] int64
    Att_Out,  # [BS, H_Q, N_SPLITS, L_EXT, Dv]  fp32
    Att_Lse,  # [BS, H_Q, N_SPLITS, L_EXT]      fp32
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,  # padded power-of-2 row tile (>= real l_ext)
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    # fp4_mx_block16 pool: K/V are packed nibbles, scales one byte per 16 values
    K_Scale=None,
    V_Scale=None,
    stride_ks_bs=0,
    stride_vs_bs=0,
    KV_FP4: tl.constexpr = False,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)

    # real number of draft query tokens for this seq
    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

    # split sizing identical to the decode kernel
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, N_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([L_EXT], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([L_EXT], dtype=tl.float32)
    acc = tl.zeros([L_EXT, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # q tile: [L_EXT, D]
        offs_q = (
            (cur_q_start + offs_l)[:, None] * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q = tl.load(
            Q + offs_q,
            mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        if KV_FP4:
            q_even, q_odd = load_q_fp4_pair(
                Q, (cur_q_start + offs_l) * stride_qbs + cur_head * stride_qh, mask_l, BLOCK_DMODEL
            )
        else:
            q_k = q.to(K_Buffer.dtype.element_ty)

        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=n_mask,
                other=0,
            )
            # K block: [D, BLOCK_N]
            if KV_FP4:
                k_pk, k_sc = load_k_fp4_packed(
                    K_Buffer, K_Scale, kv_loc, n_mask, cur_kv_head,
                    stride_buf_kbs, stride_buf_kh, stride_ks_bs, BLOCK_DMODEL, BLOCK_N,
                )
                qk = tl.dot(q_even, fp4_even(k_pk, k_sc))
                qk += tl.dot(q_odd, fp4_odd(k_pk, k_sc))
            else:
                offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
                k = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=(offs_d[:, None] < HEAD_DIM) & n_mask[None, :],
                    other=0.0,
                )
                qk = tl.dot(q_k, k)  # [L_EXT, BLOCK_N]
            qk *= sm_scale * k_scale  # fp8 dequant of prefix K (k_scale==1 if bf16)
            # NO causal mask: full prefix is visible to all draft tokens.
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # V block: [BLOCK_N, Dv]
            if KV_FP4:
                v = load_v_fp4(
                    V_Buffer, V_Scale, kv_loc, n_mask, cur_kv_head,
                    stride_buf_vbs, stride_buf_vh, stride_vs_bs, BLOCK_DV, BLOCK_N,
                )
            else:
                offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=n_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
                    other=0.0,
                )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # fp8 dequant of prefix V: scale the accumulated (pre-normalised) output.
        acc *= v_scale

        offs_o = (
            cur_batch * stride_ob
            + cur_head * stride_oh
            + split_kv_id * stride_os
            + offs_l[:, None] * stride_ol
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_o,
            acc / e_sum[:, None],
            mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        )

        offs_lse = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + split_kv_id * stride_ls
            + offs_l
        )
        tl.store(Att_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_l)
    else:
        # split did not run: write a sentinel lse so stage2 can ignore it.
        offs_lse = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + split_kv_id * stride_ls
            + offs_l
        )
        tl.store(
            Att_Lse + offs_lse,
            tl.zeros([L_EXT], tl.float32) - float("inf"),
            mask=mask_l,
        )


@triton.jit
def _verify_combine_stage2(
    Att_Out,  # [BS, H_Q, N_SPLITS, L_EXT, Dv]  fp32
    Att_Lse,  # [BS, H_Q, N_SPLITS, L_EXT]      fp32
    Q,  # [extend_tokens, H_Q, D]   (draft queries)
    K_Extend,  # [extend_tokens, H_KV, D]
    V_Extend,  # [extend_tokens, H_KV, Dv]
    O_Out,  # [extend_tokens, H_Q, Dv]  (final, written)
    sm_scale,
    qo_indptr,  # [BS+1] int32
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_qbs,
    stride_qh,
    stride_kebs,
    stride_keh,
    stride_vebs,
    stride_veh,
    stride_oobs,
    stride_ooh,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    # partials written by the grouped decode kernel: one row per (kv_head, l, g),
    # r = (h // G) * ROW_KV + l * G + h % G, ROW_KV >= L_EXT * G padded to the
    # decode kernel's head tile; 0 keeps the [b, h, s, l] layout
    ROW_GROUP: tl.constexpr = 0,
    ROW_KV: tl.constexpr = 0,
    stride_ll=1,
    num_kv_splits=None,  # [BS] int32 per-batch split count; None: all N_SPLITS written
    kv_indptr=None,  # [BS+1] int32; None: every sequence has a prefix
    BLOCK_S: tl.constexpr = 8,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num
    if ROW_GROUP > 0:
        head_row = (cur_head // ROW_GROUP) * ROW_KV + cur_head % ROW_GROUP
    else:
        head_row = cur_head

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext
    if num_kv_splits is not None:
        # the grouped stage1 skips empty splits and leaves their partials stale
        n_splits = tl.load(num_kv_splits + cur_batch)
    else:
        n_splits = N_SPLITS
    if kv_indptr is not None:
        has_prefix = (tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)) > 0
    else:
        has_prefix = True

    # ---- (a) combine prefix splits (online logsumexp over BLOCK_S chunks) ----
    # Only the first n_splits partials are read: at short context the grouped
    # stage1 fills one or two of MAX_SPLITS.
    m_p = tl.zeros([L_EXT], dtype=tl.float32) - float("inf")
    denom_p = tl.zeros([L_EXT], dtype=tl.float32)
    o_acc = tl.zeros([L_EXT, BLOCK_DV], dtype=tl.float32)
    for s0 in range(0, n_splits, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = (offs_s < n_splits) & has_prefix
        offs_lse = (
            cur_batch * stride_lb
            + head_row * stride_lh
            + offs_s[:, None] * stride_ls
            + offs_l[None, :] * stride_ll
        )
        lse = tl.load(
            offs_lse + Att_Lse,
            mask=mask_s[:, None] & mask_l[None, :],
            other=float("-inf"),
        )  # [BLOCK_S, L_EXT]
        m_new = tl.maximum(m_p, tl.max(lse, 0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        w = tl.exp(lse - m_safe[None, :])  # -inf -> 0
        alpha = tl.exp(m_p - m_safe)
        offs_ao = (
            cur_batch * stride_ob
            + head_row * stride_oh
            + offs_s[:, None, None] * stride_os
            + offs_l[None, :, None] * stride_ol
            + offs_dv[None, None, :]
        )
        # stale partials past n_splits are finite (buffers zeroed at allocation,
        # stage1 writes finite values) and get w == 0; a 3-D mask would defeat
        # the vector loads
        ao = tl.load(
            offs_ao + Att_Out,
            mask=mask_l[None, :, None] & (offs_dv[None, None, :] < V_HEAD_DIM),
            other=0.0,
        )  # [BLOCK_S, L_EXT, Dv]
        o_acc = o_acc * alpha[:, None] + tl.sum(ao * w[:, :, None], 0)
        denom_p = denom_p * alpha + tl.sum(w, 0)
        m_p = m_new
    o_prefix = tl.where(denom_p[:, None] > 0, o_acc / denom_p[:, None], 0.0)
    lse_prefix = tl.where(denom_p > 0, m_p + tl.log(denom_p), float("-inf"))  # [L_EXT]

    # ---- (b) draft-draft causal attention (L_EXT x L_EXT) -----------------
    # load draft queries [L_EXT, D], draft K/V [L_EXT, D]/[L_EXT, Dv]
    offs_q = (
        (cur_q_start + offs_l)[:, None] * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q + offs_q,
        mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    offs_ke = (
        (cur_q_start + offs_l)[:, None] * stride_kebs
        + cur_kv_head * stride_keh
        + offs_d[None, :]
    )
    ke = tl.load(
        K_Extend + offs_ke,
        mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    offs_ve = (
        (cur_q_start + offs_l)[:, None] * stride_vebs
        + cur_kv_head * stride_veh
        + offs_dv[None, :]
    )
    ve = tl.load(
        V_Extend + offs_ve,
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    # scores[i,j] = q_i . k_j  (i query, j key)  -> [L_EXT, L_EXT]
    qk = tl.sum(q[:, None, :] * ke[None, :, :], 2) * sm_scale
    # causal among drafts: query i sees key j iff j <= i, and both valid
    causal = (offs_l[None, :] <= offs_l[:, None]) & mask_l[None, :] & mask_l[:, None]
    qk = tl.where(causal, qk, float("-inf"))
    m_d = tl.max(qk, 1)  # [L_EXT]
    pd = tl.exp(qk - m_d[:, None])  # [L_EXT, L_EXT]
    denom_d = tl.sum(pd, 1)  # [L_EXT]
    o_draft = tl.sum(pd[:, :, None] * ve[None, :, :], 1)  # [L_EXT, Dv]
    o_draft = o_draft / denom_d[:, None]
    lse_draft = m_d + tl.log(denom_d)  # [L_EXT]

    # ---- (c) final LSE merge (prefix vs draft) ----------------------------
    m = tl.maximum(lse_prefix, lse_draft)
    wp = tl.exp(lse_prefix - m)
    wd = tl.exp(lse_draft - m)
    o = (o_prefix * wp[:, None] + o_draft * wd[:, None]) / (wp + wd)[:, None]

    offs_oo = (
        (cur_q_start + offs_l)[:, None] * stride_oobs
        + cur_head * stride_ooh
        + offs_dv[None, :]
    )
    tl.store(
        O_Out + offs_oo,
        o.to(O_Out.dtype.element_ty),
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
    )


class VerifySplitKV:
    """Pre-allocates scratch buffers for a problem shape and runs the split-KV
    verify attention end to end (two Triton launches: prefix split-KV + fused
    combine/draft/merge). Buffers are sized by ``max_bs`` (constant for the
    server lifetime) and reused for every batch size <= max_bs, so their
    addresses stay fixed (CUDA/HIP-graph safe) and GPU memory does not grow per
    batch size. The kernel grid uses the actual per-call bs (<= max_bs)."""

    def __init__(
        self,
        max_bs,
        h_q,
        h_kv,
        head_dim,
        v_head_dim,
        l_ext,
        device="cuda",
        n_splits=DEFAULT_N_SPLITS,
        block_n=DEFAULT_BLOCK_N,
        num_warps=DEFAULT_NUM_WARPS,
    ):
        self.h_q = h_q
        self.h_kv = h_kv
        self.group = h_q // h_kv
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.l_ext = l_ext  # real draft tokens per seq (fixed == 4)
        self.l_pad = triton.next_power_of_2(l_ext)
        self.device = device
        self.n_splits = n_splits
        self.block_n = block_n
        self.num_warps = num_warps
        self._alloc(max_bs)

    def _alloc(self, max_bs):
        # prefix split partials (fp32), sized for the maximum batch size.
        self.max_bs = max_bs
        self.att_out = torch.empty(
            (max_bs, self.h_q, self.n_splits, self.l_pad, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.att_lse = torch.empty(
            (max_bs, self.h_q, self.n_splits, self.l_pad),
            dtype=torch.float32,
            device=self.device,
        )

    def grow_buffers(self, max_bs):
        if max_bs > self.max_bs:
            self._alloc(max_bs)

    def _run_prefix_kernel(
        self,
        bs,
        q_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
        kv_fp4_scales=None,
    ):
        grid = (bs, self.h_q, self.n_splits)
        kv_fp4 = kv_fp4_scales is not None
        _verify_prefix_stage1[grid](
            q_extend,
            k_buffer,
            v_buffer,
            sm_scale,
            k_scale,
            v_scale,
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.att_out,
            self.att_lse,
            q_extend.stride(0),
            q_extend.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            num_warps=self.num_warps,
            num_stages=1,
            K_Scale=kv_fp4_scales[0] if kv_fp4 else None,
            V_Scale=kv_fp4_scales[1] if kv_fp4 else None,
            stride_ks_bs=kv_fp4_scales[0].stride(0) if kv_fp4 else 0,
            stride_vs_bs=kv_fp4_scales[1].stride(0) if kv_fp4 else 0,
            KV_FP4=kv_fp4,
            **_AMD_LAUNCH_KWARGS,
        )

    def _run_combine_kernel(
        self, bs, q_extend, k_extend, v_extend, o_out, qo_indptr, sm_scale
    ):
        grid = (bs, self.h_q)
        _verify_combine_stage2[grid](
            self.att_out,
            self.att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            sm_scale,
            qo_indptr,
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            o_out.stride(0),
            o_out.stride(1),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            num_warps=1,
            num_stages=1,
        )

    def __call__(
        self,
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        o_out=None,
        k_scale=1.0,
        v_scale=1.0,
        kv_fp4_scales=None,
    ):
        if o_out is None:
            o_out = torch.empty(
                (q_extend.shape[0], self.h_q, self.v_head_dim),
                dtype=q_extend.dtype,
                device=q_extend.device,
            )
        # actual batch size for this call (<= max_bs); the grid uses it while the
        # scratch buffers stay max_bs-sized (only the first bs slices are touched).
        bs = qo_indptr.shape[0] - 1
        # 1. prefix split-KV
        self._run_prefix_kernel(
            bs,
            q_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            k_scale,
            v_scale,
            kv_fp4_scales=kv_fp4_scales,
        )
        # 2+3+4. fused combine + draft-draft + merge
        self._run_combine_kernel(
            bs,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            qo_indptr,
            sm_scale,
        )
        return o_out


@triton.jit
def _grouped_num_splits(
    kv_indptr, out, bs, MAX_SPLITS: tl.constexpr, TOKENS_PER_SPLIT: tl.constexpr, BLOCK: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    m = offs < bs
    lens = tl.load(kv_indptr + offs + 1, mask=m, other=0) - tl.load(kv_indptr + offs, mask=m, other=0)
    n = tl.minimum(tl.maximum(tl.cdiv(lens, TOKENS_PER_SPLIT), 1), MAX_SPLITS)
    # stage 1 rounds splits up to MIN_BLOCK_KV and skips the ones that leaves empty, while
    # stage 2 reads the first n partials: count only the non-empty splits
    per_split = tl.maximum(tl.cdiv(tl.cdiv(lens, n), MIN_BLOCK_KV) * MIN_BLOCK_KV, MIN_BLOCK_KV)
    n = tl.maximum(tl.cdiv(lens, per_split), 1)
    tl.store(out + offs, n, mask=m)


class VerifyGroupedKV:
    """Verify attention with the prefix stage on the grouped decode kernel: one
    program per (kv head, split) serves every q head and draft token of that
    kv head, so K/V are read once instead of once per q head (3.5x faster than
    VerifySplitKV at GQA 40/4 on gfx1201, fp8 and fp4 alike). Rows are ordered
    (kv_head, l, g); _verify_combine_stage2 maps them back with ROW_GROUP."""

    MAX_SPLITS = 96
    # per-sequence split count = cdiv(prefix, TOKENS_PER_SPLIT) capped at MAX_SPLITS,
    # computed on device from kv_indptr so a captured graph adapts to the real
    # length (a static choice would bake the capture-time padded shape in).
    # Measured on gfx1201 (64 CUs) from 512 to 200K prefix tokens: 256-token splits fill
    # the device from 2K up; fp8 is fastest at 32 splits, fp4 (more ALU per byte) at 64.
    TOKENS_PER_SPLIT = 256
    SPLIT_CAP_FP8 = 32
    SPLIT_CAP_FP4_TRITON = 64

    def __init__(self, max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, device="cuda"):
        self.h_q, self.h_kv, self.group = h_q, h_kv, h_q // h_kv
        self.head_dim, self.v_head_dim = head_dim, v_head_dim
        self.l_pad = triton.next_power_of_2(l_ext)
        # rows per kv head, padded to the grouped decode kernel's 16-row head
        # tile (it assumes kv_group_num > 16 is a multiple of 16); pad rows are
        # zero queries the combine stage never reads
        self.rows_used = self.l_pad * self.group
        self.rows_kv = triton.cdiv(self.rows_used, 16) * 16 if self.rows_used > 16 else self.rows_used
        self.rows = h_kv * self.rows_kv
        # one head tile per kv head when the rows fit 32, so K/V are read (and fp4-unpacked)
        # once per kv head; the grouped kernel needs rows_kv to be a multiple of the tile
        self.block_h = 32 if self.rows_kv % 32 == 0 else 16
        self.device = device
        self.max_bs = max_bs
        self.qg = torch.zeros((max_bs, h_kv, self.rows_kv, head_dim), dtype=torch.bfloat16, device=device)
        self.att_out = torch.zeros(
            (max_bs, self.rows, self.MAX_SPLITS, v_head_dim), dtype=torch.float32, device=device
        )
        self.att_lse = torch.empty((max_bs, self.rows, self.MAX_SPLITS), dtype=torch.float32, device=device)
        self.num_kv_splits = torch.full((max_bs,), self.MAX_SPLITS, dtype=torch.int32, device=device)
        self.offs_l = torch.arange(self.l_pad, device=device, dtype=torch.int32)

    def _gather_rows(self, q_extend, qo_indptr, bs):
        # [bs, l_pad] source rows; missing drafts re-read row 0 of the sequence
        # (the combine stage masks them out). Device-only, static shapes.
        start = qo_indptr[:bs, None]
        idx = start + self.offs_l[None, :]
        idx = torch.where(idx < qo_indptr[1 : bs + 1, None], idx, start)
        qb = q_extend[idx.reshape(-1)].view(bs, self.l_pad, self.h_kv, self.group, self.head_dim)
        if self.qg.dtype != q_extend.dtype:
            self.qg = torch.zeros_like(self.qg, dtype=q_extend.dtype)
        self.qg[:bs, :, : self.rows_used].copy_(
            qb.permute(0, 2, 1, 3, 4).reshape(bs, self.h_kv, self.rows_used, self.head_dim)
        )
        return self.qg[:bs].view(bs, self.rows, self.head_dim)

    def __call__(
        self, q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
        sm_scale, o_out=None, k_scale=1.0, v_scale=1.0, kv_fp4_scales=None,
    ):
        from sglang.kernels.ops.attention.decode_attention import _decode_grouped_att_m_fwd

        if o_out is None:
            o_out = torch.empty((q_extend.shape[0], self.h_q, self.v_head_dim), dtype=q_extend.dtype, device=q_extend.device)
        bs = qo_indptr.shape[0] - 1
        # HIP stage 1 (fp4 pool, head_dim 256): measured best at 96 splits of >= 256 tokens
        use_hip = (
            kv_fp4_scales is not None
            and _fp4_attn_hip is not None
            and self.head_dim == 256
            and self.v_head_dim == 256
            and self.rows_kv <= 32
            and q_extend.dtype == torch.bfloat16
            and kv_indices.dtype == torch.int64
            and kv_indptr.dtype == torch.int32
        )
        if kv_fp4_scales is None:
            split_cap = self.SPLIT_CAP_FP8
        else:
            split_cap = self.MAX_SPLITS if use_hip else self.SPLIT_CAP_FP4_TRITON
        _grouped_num_splits[(1,)](
            kv_indptr, self.num_kv_splits, bs, MAX_SPLITS=split_cap,
            TOKENS_PER_SPLIT=self.TOKENS_PER_SPLIT, BLOCK=triton.next_power_of_2(max(bs, 1)),
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
        )
        qg = self._gather_rows(q_extend, qo_indptr, bs)
        if use_hip:
            _fp4_attn_hip.fp4_attn_stage1(
                qg, k_buffer, v_buffer, kv_fp4_scales[0], kv_fp4_scales[1], kv_indptr, kv_indices,
                self.num_kv_splits, self.att_out[:bs], self.att_lse[:bs], self.rows_kv, self.rows_used,
                sm_scale * k_scale, _MIN_BLOCK_KV,
            )
        else:
            _decode_grouped_att_m_fwd(
                qg, k_buffer, v_buffer, self.att_out, self.att_lse, kv_indptr, kv_indices,
                self.num_kv_splits, self.MAX_SPLITS, sm_scale * k_scale, 0.0,
                kv_fp4_scales=kv_fp4_scales, block_h=self.block_h,
            )
        if v_scale != 1.0:
            self.att_out[:bs].mul_(v_scale)
        grid = (bs, self.h_q)
        _verify_combine_stage2[grid](
            self.att_out, self.att_lse, q_extend, k_extend, v_extend, o_out, sm_scale, qo_indptr,
            self.att_out.stride(0), self.att_out.stride(1), self.att_out.stride(2), self.group * self.att_out.stride(1),
            self.att_lse.stride(0), self.att_lse.stride(1), self.att_lse.stride(2),
            q_extend.stride(0), q_extend.stride(1), k_extend.stride(0), k_extend.stride(1),
            v_extend.stride(0), v_extend.stride(1), o_out.stride(0), o_out.stride(1),
            kv_group_num=self.group, N_SPLITS=self.MAX_SPLITS, L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim, V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim), BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            ROW_GROUP=self.group, ROW_KV=self.rows_kv, stride_ll=self.group * self.att_lse.stride(1),
            num_kv_splits=self.num_kv_splits, kv_indptr=kv_indptr, BLOCK_S=8,
            num_warps=1, num_stages=1,
        )
        return o_out


_VG_CACHE = {}


def _get_vg(max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, device):
    key = (h_q, h_kv, head_dim, v_head_dim, l_ext, str(device))
    vg = _VG_CACHE.get(key)
    if vg is None or vg.max_bs < max_bs:
        vg = VerifyGroupedKV(max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, device)
        _VG_CACHE[key] = vg
    return vg


def verify_grouped_fwd(
    q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
    custom_mask, is_causal, mask_indptr, max_len_extend, k_scale, v_scale, sm_scale=None,
    logit_cap=0.0, skip_prefix_custom_mask=True, sliding_window_size=-1, sinks=None,
    window_kv_offsets=None, xai_temperature_len=-1, max_bs=None, kv_fp4_scales=None,
):
    """verify_splitkv_fwd with the prefix stage on the grouped decode kernel.
    Same contract: True if it ran, False to fall back to extend_attention_fwd."""
    if not can_handle(
        q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
        custom_mask, is_causal, mask_indptr, max_len_extend, sliding_window_size=sliding_window_size,
        sinks=sinks, logit_cap=logit_cap, xai_temperature_len=xai_temperature_len,
        kv_fp4=kv_fp4_scales is not None,
    ):
        return False
    h_q, h_kv = q_extend.shape[1], k_extend.shape[1]
    if h_q == h_kv:
        return False  # grouped decode kernel wants GQA
    head_dim, v_head_dim = q_extend.shape[2], v_extend.shape[2]
    if v_head_dim != triton.next_power_of_2(v_head_dim):
        return False
    bs = qo_indptr.shape[0] - 1
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    try:
        k_scale = float(k_scale)
    except (TypeError, ValueError):
        k_scale = 1.0
    try:
        v_scale = float(v_scale)
    except (TypeError, ValueError):
        v_scale = 1.0
    if max_bs is None or max_bs < bs:
        max_bs = bs
    vg = _get_vg(max_bs, h_q, h_kv, head_dim, v_head_dim, int(max_len_extend), q_extend.device)
    vg(
        q_extend, k_extend.contiguous(), v_extend.contiguous(), k_buffer, v_buffer, qo_indptr, kv_indptr,
        kv_indices, sm_scale, o_out=o_extend, k_scale=k_scale, v_scale=v_scale, kv_fp4_scales=kv_fp4_scales,
    )
    return True


# ---------------------------------------------------------------------------
# Live-server dispatch entry.
# ---------------------------------------------------------------------------
# Cache one VerifySplitKV instance per (h_q, h_kv, head_dim, v_head_dim, l_ext,
# device, n_splits) shape -- NOT keyed on the dynamic batch size. Buffers are
# sized by the stable max_bs (grown only if a larger one is ever requested), so
# a single instance serves every batch size: addresses stay fixed (graph-safe)
# and GPU memory does not grow per batch size.
_VK_CACHE = {}


def _get_vk(
    max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, device, n_splits=DEFAULT_N_SPLITS, kv_fp4=False
):
    key = (h_q, h_kv, head_dim, v_head_dim, l_ext, str(device), n_splits, kv_fp4)
    vk = _VK_CACHE.get(key)
    if vk is None:
        block_n, num_warps = block_config(head_dim)
        if kv_fp4:
            # packed nibbles decode to bf16 halves in registers: a narrower tile
            # keeps the 256-VGPR budget without spills (see decode_attention.py)
            block_n = int(os.environ.get("SGLANG_KV4_VERIFY_BLOCK_N", 16))
            num_warps = int(os.environ.get("SGLANG_KV4_VERIFY_NUM_WARPS", 4))
        vk = VerifySplitKV(
            max_bs,
            h_q,
            h_kv,
            head_dim,
            v_head_dim,
            l_ext,
            device=device,
            n_splits=n_splits,
            block_n=block_n,
            num_warps=num_warps,
        )
        _VK_CACHE[key] = vk
    else:
        vk.grow_buffers(max_bs)
    return vk


def can_handle(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    sliding_window_size=-1,
    sinks=None,
    logit_cap=0.0,
    xai_temperature_len=-1,
    kv_fp4=False,
):
    """Return True iff the split-KV verify path can serve this exact problem
    with the same result as extend_attention_fwd. Conservative: anything not
    explicitly handled -> False -> caller falls back to the baseline.

    IMPORTANT: ``custom_mask`` is intentionally NOT inspected (its values can't
    be read inside a captured HIP graph without a host sync). The kernel always
    computes pure-causal attention, which equals the tree mask ONLY at
    speculative topk == 1. The caller therefore MUST gate enablement on topk == 1
    (TritonAttnBackend does: ``use_verify_splitkv = ... and self.topk == 1``).
    At topk > 1 the tree is not causal and this path must stay disabled."""
    # No exotic features.
    if sinks is not None:
        return False
    if sliding_window_size is not None and sliding_window_size > 0:
        return False
    if logit_cap and logit_cap > 0:
        return False
    if xai_temperature_len is not None and xai_temperature_len > 0:
        return False
    if not is_causal:
        return False
    # q layout must be [tokens, H_Q, D]; head dims handled by power-of-2 pad.
    if q_extend.dim() != 3 or k_extend.dim() != 3 or v_extend.dim() != 3:
        return False
    # GQA group must divide evenly.
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    if h_kv == 0 or h_q % h_kv != 0:
        return False
    # head dims must match buffers.
    if k_buffer.shape[1] != h_kv or v_buffer.shape[1] != h_kv:
        return False
    if q_extend.shape[2] != k_extend.shape[2]:
        return False
    # packed fp4 nibbles: the pool's last dim is half the head dim
    pack = 2 if kv_fp4 else 1
    if q_extend.shape[2] != k_buffer.shape[2] * pack:
        return False
    if v_extend.shape[2] != v_buffer.shape[2] * pack:
        return False
    # MLA (head_dim != v_head_dim, e.g. DeepSeek 576 vs 512) uses a shared
    # latent KV cache and an absorbed-attention layout the split-KV verify
    # kernel is not built for; it GPU-faults on that shape. Fall back to
    # extend_attention_fwd, which handles MLA correctly.
    if q_extend.shape[2] != v_extend.shape[2]:
        return False
    # NOTE: must NOT read any tensor *values* here (no .item()/.cpu()): the
    # target-verify step runs inside a captured CUDA/HIP graph, where a
    # device->host sync raises hipErrorStreamCaptureUnsupported. We therefore
    # gate purely on static shapes/dtypes/python scalars.
    bs = qo_indptr.shape[0] - 1
    if bs < 1:
        return False
    # max_len_extend must be a known positive python int (it is the static
    # server_args.speculative_num_draft_tokens for the verify path). For
    # topk=1 the per-seq extend len is constant == num_draft_tokens ==
    # max_len_extend by construction of qo_indptr (arange with that step), so
    # the L_EXT row-tile mask is exactly right and the tree custom_mask equals
    # causal -- no value inspection required.
    try:
        mle = int(max_len_extend)
    except (TypeError, ValueError):
        return False
    if mle < 1:
        return False
    # The packed extend tensor must hold exactly bs * max_len_extend rows
    # (constant extend len). This is a pure shape check (no sync) and rejects
    # any ragged/variable-extend batch -> falls back to the baseline.
    if q_extend.shape[0] != bs * mle:
        return False
    return True


def verify_splitkv_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    max_bs=None,
    kv_fp4_scales=None,
):
    """Drop-in for extend_attention_fwd on the EAGLE target-verify (topk=1)
    shape. Returns True if it ran (o_extend written), False if the case is
    unsupported and the caller must fall back to extend_attention_fwd.

    ``max_bs`` (optional) is the stable maximum batch size used to size the
    cached scratch buffers; the backend passes its req_to_token_pool size. If
    omitted it defaults to this call's bs.

    Arg order mirrors extend_attention_fwd exactly so the call site is a
    one-line swap.
    """
    if not can_handle(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_len_extend,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        kv_fp4=kv_fp4_scales is not None,
    ):
        return False

    bs = qo_indptr.shape[0] - 1
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    head_dim = q_extend.shape[2]
    v_head_dim = v_extend.shape[2]
    l_ext = int(max_len_extend)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    # k_scale/v_scale may be float or 0-d tensor; coerce to python float.
    try:
        k_scale = float(k_scale)
    except (TypeError, ValueError):
        k_scale = 1.0
    try:
        v_scale = float(v_scale)
    except (TypeError, ValueError):
        v_scale = 1.0

    # Adaptive split count from the average prefix length. This is a
    # pure-shape derivation (kv_indices.shape[0] / bs) -- no device->host sync,
    # so it is safe inside a captured HIP graph. The whole batch shares one
    # N_SPLITS (the grid dim must be a launch constexpr); the per-split kernel
    # logic still clamps each split's [start,end) to that seq's real length, so
    # mixed-length batches stay correct -- shorter seqs simply write fewer
    # active splits (the rest emit the -inf lse sentinel, ignored in stage2).
    avg_seqlen = kv_indices.shape[0] / max(1, bs)
    n_splits = choose_n_splits(avg_seqlen)

    # Size scratch by the stable max_bs (backend passes req_to_token_pool size);
    # fall back to this call's bs if not provided / smaller.
    if max_bs is None or max_bs < bs:
        max_bs = bs
    vk = _get_vk(
        max_bs,
        h_q,
        h_kv,
        head_dim,
        v_head_dim,
        l_ext,
        q_extend.device,
        n_splits=n_splits,
        kv_fp4=kv_fp4_scales is not None,
    )
    vk(
        q_extend,
        k_extend.contiguous(),
        v_extend.contiguous(),
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        o_out=o_extend,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_fp4_scales=kv_fp4_scales,
    )

    return True
