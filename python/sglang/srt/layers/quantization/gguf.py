# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/ab3e80042eac24dd362408e6d63ad98768046359/vllm/model_executor/layers/quantization/gguf.py
from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, List, Optional

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUUnquantMoEMethod,
)
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()
_is_npu = is_npu()

if _is_cuda or _is_hip:
    # gfx1201: ggml ops come from gguf-kernels-gfx1201.sh; moe_sum is not in the
    # ROCm extension (only the MoE method needs it).
    from sgl_kernel import moe_align_block_size

    try:
        from sgl_kernel import moe_sum
    except ImportError:  # ROCm build
        moe_sum = None
    if _is_hip:
        # the Python wrapper imports fine but torch.ops.sgl_kernel.moe_sum is not
        # registered in the ROCm extension; use the torch.sum fallback below
        moe_sum = None
        # sgl_kernel's ROCm moe_align_block_size takes preallocated buffers; the
        # 3-tuple helper is what the code below expects
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size,
        )
        try:
            from sglang.srt.layers.quantization.gguf_moe_triton import (
                fused_moe_gguf_triton as _fused_moe_gguf_triton,
            )
        except Exception:  # pragma: no cover
            _fused_moe_gguf_triton = None
    from sgl_kernel.quantization import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    from sglang.kernels.ops.activation.activation import gelu_and_mul, silu_and_mul
elif _is_musa:
    from sgl_kernel import gelu_and_mul, moe_align_block_size, moe_sum, silu_and_mul
    from sgl_kernel.quantization import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )
elif _is_npu:
    from gguf import dequantize as gguf_dequantize
else:
    warnings.warn(f"Only CUDA, HIP, MUSA and NPU support GGUF quantization currently.")

logger = logging.getLogger(__name__)


def _ordered_gguf_shard_ids(shard_ids: list) -> list:
    """Return checkpoint shards in the fused layer's logical output order."""
    if len(shard_ids) == 3 and set(shard_ids) == {"q", "k", "v"}:
        return ["q", "k", "v"]
    if all(isinstance(shard_id, int) for shard_id in shard_ids) and set(
        shard_ids
    ) == set(range(len(shard_ids))):
        return sorted(shard_ids)
    if all(isinstance(s, (int, tuple)) for s in shard_ids):
        return sorted(shard_ids, key=lambda s: s[0] if isinstance(s, tuple) else s)
    return list(shard_ids)


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, modules_to_not_convert: list[str] | None = None) -> None:
        super().__init__()
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60 if not _is_musa else 21

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GGUFConfig:
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            if _is_npu:
                return GGUFLinearAscendMethod(self)
            method = GGUFLinearMethod(self)
            method.prefix = prefix
            return method
        elif isinstance(layer, VocabParallelEmbedding):
            if is_layer_skipped_gguf(prefix, self.modules_to_not_convert):
                return None  # -> UnquantizedEmbeddingMethod (vision pos_embed etc.)
            if _is_npu:
                return GGUFEmbeddingAscendMethod(self)
            return GGUFEmbeddingMethod(self)
        elif isinstance(layer, FusedMoE):
            if _is_npu:
                return GGUFMoEAscendMethod(self)
            return GGUFMoEMethod(self)
        return None


def is_layer_skipped_gguf(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
if _is_hip:
    # padded Q6_K (1014): served by mmvq (case 1014), dequant (case 1014) and the fused GEMM
    DEQUANT_TYPES = DEQUANT_TYPES | {1014}
    MMVQ_QUANT_TYPES = MMVQ_QUANT_TYPES | {1014}
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES
if _is_hip:
    # gfx1201 measurements (sglang-gfx1201/test_gguf_kernels.py): the MI300-tuned
    # mmq tiles are 2-5x slower than dequantize + hipBLASLt at M=128..2048 and
    # the Q4_K/Q5_K variants are less accurate than mmvq. Prefill goes through
    # the dequant path; decode (M <= mmvq_safe) stays on mmvq.
    MMQ_QUANT_TYPES = set()


def _gguf_sizes(qweight_type: int):
    """(block_size, type_size), aware of the gfx1201 padded Q6_K layout (1014)."""
    if qweight_type == 1014:
        return 256, 224
    return gguf.GGML_QUANT_SIZES[qweight_type]


def dequantize_gguf_weight(
    qweight: torch.Tensor, qweight_type: int, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize a packed GGUF matrix using its inferred logical shape."""
    block_size, type_size = _gguf_sizes(qweight_type)
    shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
    return ggml_dequantize(qweight, qweight_type, *shape, dtype)


_DEQUANT_CHUNK_BYTES = 256 << 20

# gfx1201 fused path (see gguf_triton.py). Q6_K padded = 1014.
_GGUF_Q6_K_PAD = 1014


def _gguf_trim_host(where: str) -> None:
    # Give freed host staging back to the OS and log RssAnon before/after: the
    # expert shards are staged on the CPU during load, and the scheduler was
    # measured holding ~20 GB anonymous RSS at idle afterwards.
    import ctypes, gc, logging
    def rss():
        try:
            for line in open("/proc/self/status"):
                if line.startswith("RssAnon"):
                    return int(line.split()[1]) // 1024
        except OSError:
            pass
        return -1
    before = rss(); gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    logging.getLogger(__name__).info("gguf host trim %s: RssAnon %d MB -> %d MB", where, before, rss())
if _is_hip:
    try:
        from sglang.srt.layers.quantization.gguf_triton import (
            FUSED_TYPES as _GGUF_FUSED_TYPES,
            gguf_gemm_triton as _gguf_gemm_triton,
        )
    except Exception:  # pragma: no cover
        _GGUF_FUSED_TYPES, _gguf_gemm_triton = set(), None
    import os as _os

    _GGUF_FUSED_MIN_M = int(_os.environ.get("SGLANG_GGUF_FUSED_MIN_M", "4"))
    _GGUF_FUSED_MAX_M = int(_os.environ.get("SGLANG_GGUF_FUSED_MAX_M", "64"))
else:
    _GGUF_FUSED_TYPES, _gguf_gemm_triton = set(), None
# gfx1201 fp8 prefill (gguf-fp8-prefill-gfx1201.sh)
_GGUF_FP8 = _is_hip and os.environ.get("SGLANG_GGUF_FP8_PREFILL", "0") == "1"
_gguf_fp8 = None
if _GGUF_FP8:
    try:
        from sglang.srt.layers.quantization import gguf_fp8 as _gguf_fp8
    except Exception as _e:  # pragma: no cover
        logger.warning("SGLANG_GGUF_FP8_PREFILL=1 but gguf_fp8 failed to import: %s", _e)
        _GGUF_FP8 = False
_GGUF_FP8_MIN_M = int(os.environ.get("SGLANG_GGUF_FP8_MIN_M", "128"))
_GGUF_FP8_MIN_N = int(os.environ.get("SGLANG_GGUF_FP8_MIN_N", "256"))
# lm_head only sees prefill-sized M for input-logprob requests; keep those on bf16
_GGUF_FP8_SKIP = [t for t in os.environ.get("SGLANG_GGUF_FP8_SKIP", "lm_head").split(",") if t]
_GGUF_FP8_PER_TOKEN = os.environ.get("SGLANG_GGUF_FP8_ACT", "token") != "tensor"
_GGUF_FP8_CHECK = os.environ.get("SGLANG_GGUF_FP8_CHECK", "0") == "1"
# gfx1201 fp8 MoE (moe-fp8-gfx1201.sh)
_MOE_FP8 = _is_hip and os.environ.get("SGLANG_GGUF_MOE_FP8", "0") == "1"
_gguf_moe_fp8 = None
if _MOE_FP8:
    try:
        from sglang.srt.layers.quantization import gguf_moe_fp8 as _gguf_moe_fp8
    except Exception as _e:  # pragma: no cover
        logger.warning("SGLANG_GGUF_MOE_FP8=1 but gguf_moe_fp8 failed to import: %s", _e)
        _MOE_FP8 = False
_MOE_FP8_MIN_TOKENS = int(os.environ.get("SGLANG_GGUF_MOE_FP8_MIN_TOKENS", "9"))
# gfx1201 small-batch GEMV (sglang-gfx1201/kq-gemv-gfx1201.sh builds kq_gemv.so); absent -> old paths
_kq_gemv = None
if _is_hip:
    try:
        from sglang.srt.layers.quantization import kq_gemv as _kq_gemv
    except ImportError:
        _kq_gemv = None
# Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, padded Q6_K
_KQ_GEMV_TYPES = {8, 12, 13, 14, 23, 1014}


def _repack_q6_k_padded(qweight: torch.Tensor) -> torch.Tensor:
    """[N, nb*210] uint8 -> [N, nb*224] with each block zero-padded to 224 bytes."""
    n, width = qweight.shape
    nb = width // 210
    out = torch.zeros((n, nb, 224), dtype=torch.uint8, device=qweight.device)
    out[:, :, :210] = qweight.view(n, nb, 210)
    return out.view(n, nb * 224)


def _dequant_matmul_chunked(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    """x @ dequant(qweight).T with the dequantized temp bounded to ~256 MB."""
    block_size, type_size = _gguf_sizes(qweight_type)
    rows = qweight.shape[0]
    K = qweight.shape[1] // type_size * block_size
    rows_per = max(256, _DEQUANT_CHUNK_BYTES // (K * x.element_size()))
    y = torch.empty(x.shape[0], rows, dtype=x.dtype, device=x.device)
    for r0 in range(0, rows, rows_per):
        r1 = min(rows, r0 + rows_per)
        w = ggml_dequantize(qweight[r0:r1], qweight_type, r1 - r0, K, x.dtype)
        torch.matmul(x, w.T, out=y[:, r0:r1])
    return y


def fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    if _is_hip:
        mmvq_safe = 8  # gfx1201: validated for all k-quant / imatrix types in use
    elif qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if qweight.shape[0] > 5120 else 6
    # HACK: when doing chunked prefill we don't generate output tokens
    # so input to logits generator is empty which causes invalid parameter
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    # up to 8 rows: weights read once for all rows (mmvq re-reads them per row); Q4_K with
    # short rows stays on the fused Triton GEMM above 4 rows, where that measured faster
    if (
        _kq_gemv is not None
        and qweight_type in _KQ_GEMV_TYPES
        and x.dim() == 2
        and x.shape[0] <= 8
        and x.dtype == torch.bfloat16
        and x.shape[1] % 256 == 0
        and not (qweight_type == 12 and x.shape[0] > 4 and x.shape[1] < 4096)
    ):
        if x.stride(1) != 1:
            x = x.contiguous()
        return _kq_gemv.kq_gemv(x, qweight, qweight_type, qweight.shape[0], 0, 0, 0)
    if (
        _gguf_gemm_triton is not None
        and qweight_type in _GGUF_FUSED_TYPES
        and _GGUF_FUSED_MIN_M <= x.shape[0] <= _GGUF_FUSED_MAX_M
    ):
        bs, ts = _gguf_sizes(qweight_type)
        K = qweight.shape[1] // ts * bs
        return _gguf_gemm_triton(x, qweight, qweight_type, qweight.shape[0], K)
    # enable MMVQ in contiguous batching with batch_size=1
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        y = ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # Use MMQ Kernel if it's available (standard + k-quants)
    elif qweight_type in MMQ_QUANT_TYPES:
        y = ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    # If there is no available MMQ kernel, fallback to dequantize
    elif qweight_type in DEQUANT_TYPES:
        if _is_hip:
            y = _dequant_matmul_chunked(x, qweight, qweight_type)
        else:
            weight = dequantize_gguf_weight(qweight, qweight_type, x.dtype)
            y = x @ weight.T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y


def fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    def act(x: torch.Tensor):
        if activation == "silu":
            return silu_and_mul(x)
        elif activation == "gelu":
            return gelu_and_mul(x)
        raise ValueError(f"Unsupported activation: {activation}")

    out_hidden_states = torch.empty_like(x)
    # decode: one GEMV per (token, expert) pair; 8 lanes per row for single tokens and short
    # rows (measured: Qwen3.6-A3B gate_up 1 token 48 -> 28 us, down 4 tokens 61 -> 42 us)
    if (
        _kq_gemv is not None
        and qweight_type in _KQ_GEMV_TYPES
        and qweight_type2 in _KQ_GEMV_TYPES
        and x.dim() == 2
        and x.shape[0] <= 8
        and x.dtype == torch.bfloat16
        and x.shape[1] % 256 == 0
        and w2.shape[2] // _gguf_sizes(qweight_type2)[1] * _gguf_sizes(qweight_type2)[0] % 256 == 0
    ):
        num_tokens, top_k = x.shape[0], topk_ids.shape[1]
        ids = topk_ids.reshape(-1).to(torch.int32)
        pairs = ids.numel()
        lpr = 8 if pairs <= 8 or x.shape[1] <= 1024 else 32
        out = _kq_gemv.kq_gemv_moe(x.contiguous(), w1, ids, qweight_type, w1.shape[1], top_k, lpr, 0)
        out = act(out)
        lpr = 8 if pairs <= 8 or out.shape[1] <= 1024 else 32
        out = _kq_gemv.kq_gemv_moe(out, w2, ids, qweight_type2, w2.shape[1], 1, lpr, 0)
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(topk_weights.view(num_tokens, top_k, 1))
        torch.sum(out, dim=1, out=out_hidden_states)
        return out_hidden_states
    # unless we decent expert reuse we are better off running moe_vec kernel
    if (
        _is_hip
        and _fused_moe_gguf_triton is not None
        and qweight_type in _GGUF_FUSED_TYPES
        and qweight_type2 in _GGUF_FUSED_TYPES
        and x.shape[0] * topk_ids.shape[1] >= int(_os.environ.get("SGLANG_GGUF_MOE_FUSED_MIN_PAIRS", "1024"))
    ):
        return _fused_moe_gguf_triton(
            x, w1, w2, topk_weights, topk_ids, qweight_type, qweight_type2,
            moe_align_block_size, act,
        )
    if (
        qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        BLOCK_SIZE = ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, BLOCK_SIZE, E
        )
        out = ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        if moe_sum is None:  # ROCm build
            torch.sum(out, dim=1, out=out_hidden_states)
        else:
            moe_sum(out, out_hidden_states)
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)
        out = act(out)

        out = ggml_moe_a8_vec(
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        if moe_sum is None:  # ROCm build
            torch.sum(out, dim=1, out=out_hidden_states)
        else:
            moe_sum(out, out_hidden_states)
    else:
        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                expert_up = w1[ii]

                out = fused_mul_mat_gguf(inp, expert_up, qweight_type)
                out = act(out)

                expert_down = w2[ii]
                current_state = fused_mul_mat_gguf(
                    out, expert_down, qweight_type2
                ).mul_(ww)
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def apply_gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = _gguf_sizes(qweight_type)
        x_flat = x.flatten()
        assert hidden_size == qweight.shape[1] // type_size * block_size
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = ggml_dequantize(
            quant, qweight_type, hidden_size, x_flat.shape[0], dtype
        )
        return dequant.view(*x.shape, hidden_size)
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize the padded weight parameter for CUDA Graph compatibility.
        self._create_padded_weight_param(layer)
        if _is_hip and not layer.qweight.shard_id and qweight_type == WeightType.Q6_K:
            # single-shard Q6_K (attn_v, lm_head, mtp.fc): repack to 224-byte blocks
            padded = Parameter(_repack_q6_k_padded(layer.qweight.data), requires_grad=False)
            set_weight_attrs(padded, vars(layer.qweight))
            layer.register_parameter("qweight", padded)
            layer.qweight_type.weight_type = _GGUF_Q6_K_PAD
        if _GGUF_FP8:
            self._fp8_prepare(layer)

    def _fp8_prepare(self, layer: torch.nn.Module):
        """Shard views + one fp8 scale for the whole (merged) layer; leaves the layer on the
        bf16 path when any shard type has no fp8 unpack or the layer is small or skipped."""
        prefix = getattr(self, "prefix", "")
        if any(t in prefix for t in _GGUF_FP8_SKIP):
            return
        shards = self._shard_views(layer)
        if not shards or any(qt not in _gguf_fp8.FP8_TYPES for _, qt in shards):
            return
        bs, ts = _gguf_sizes(shards[0][1])
        K = shards[0][0].shape[1] // ts * bs
        if K % 256 or sum(w.shape[0] for w, _ in shards) < _GGUF_FP8_MIN_N:
            return
        amax = torch.stack([_gguf_fp8.row_amax(w, qt, K).max() for w, qt in shards]).max()
        ws = (amax.clamp_min(1e-12) / _gguf_fp8.FP8_MAX).reshape(1).float()
        layer._fp8_shards, layer._fp8_k = shards, K
        layer._fp8_ws, layer._fp8_wi = ws, (1.0 / ws).float()
        layer._fp8_prefix = prefix

    def _shard_views(self, layer: torch.nn.Module):
        qweight = layer.qweight
        if not qweight.shard_id:
            return [(qweight.data, int(layer.qweight_type.weight_type))]
        flat_map = getattr(qweight, "flat_shard_map", None)
        if flat_map is None:
            return None
        out = []
        for idx in _ordered_gguf_shard_ids(qweight.shard_id):
            off, rows, width = flat_map[idx]
            out.append((qweight.data[off : off + rows * width].view(rows, width),
                        int(layer.qweight_type.shard_weight_type[idx])))
        return out

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            # concat dim0 and pad dim1
            if _is_hip:
                # flat byte buffer, shards back-to-back, viewed in place at apply()
                ordered_shard_ids = _ordered_gguf_shard_ids(shard_id)
                flat_shard_map = {}
                cursor = 0
                shard_types = layer.qweight_type.shard_weight_type
                padded_shards = {}
                for idx in ordered_shard_ids:
                    d = data_container[shard_id_map[idx]].contiguous()
                    if shard_types.get(idx) == WeightType.Q6_K:
                        d = _repack_q6_k_padded(d)
                        padded_shards[idx] = d
                    flat_shard_map[idx] = (cursor, d.size(0), d.size(1))
                    cursor += d.numel()
                total = cursor
                flat = torch.empty(total, dtype=dtype, device=qweight.device)
                cursor = 0
                for idx in ordered_shard_ids:
                    d = padded_shards.get(idx)
                    if d is None:
                        d = data_container[shard_id_map[idx]].contiguous()
                    flat[cursor : cursor + d.numel()].copy_(d.view(-1))
                    cursor += d.numel()
                for idx in padded_shards:
                    shard_types[idx] = _GGUF_Q6_K_PAD
                qweight.data_container.clear()
                flat_param = Parameter(flat, requires_grad=False)
                set_weight_attrs(flat_param, vars(qweight))
                flat_param.shard_id = ordered_shard_ids
                set_weight_attrs(flat_param, {"flat_shard_map": flat_shard_map})
                layer.register_parameter("qweight", flat_param)
                return
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            # Pad the quantized weights to dense tensor, and create a map
            # with the location of each shard in the padded tensor.
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            # (dim0_start, dim0_end, dim1_size)
            shard_offset_map = dict[str, tuple[int, int, int]]()
            ordered_shard_ids = _ordered_gguf_shard_ids(shard_id)
            cursor = 0
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = cursor
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
                cursor = end
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            padded_param.shard_id = ordered_shard_ids
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shard_id = layer.qweight.shard_id

        if (
            _GGUF_FP8
            and x.dim() == 2
            and x.shape[0] >= _GGUF_FP8_MIN_M
            and x.dtype == torch.bfloat16
            and getattr(layer, "_fp8_ws", None) is not None
        ):
            out = _gguf_fp8.gguf_fp8_linear(
                x, layer._fp8_shards, layer._fp8_ws, layer._fp8_wi, layer._fp8_k,
                per_token=_GGUF_FP8_PER_TOKEN,
            )
            if _GGUF_FP8_CHECK and not getattr(layer, "_fp8_checked", False):
                layer._fp8_checked = True
                # row slices only: dd's card has < 1 GB free at prefill
                xs = x[:256]
                ref = torch.cat([fused_mul_mat_gguf(xs, w, qt) for w, qt in layer._fp8_shards], 1)
                err = ((out[:256].float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)).item()
                ratio = 0.0
                for r0 in range(0, x.shape[0], 256):
                    xa = x[r0 : r0 + 256].float().abs()
                    ratio = max(ratio, (xa.amax(1) / xa.median(1).values.clamp_min(1e-12)).max().item())
                logger.info("gguf fp8 check %s M=%d N=%d K=%d rel_err=%.3e in_amax/median=%.3e",
                            layer._fp8_prefix, x.shape[0], out.shape[1], layer._fp8_k, err, ratio)
            if bias is not None:
                out.add_(bias)
            return out

        if shard_id:
            # dequantize shard weights respectively
            shard_id = _ordered_gguf_shard_ids(shard_id)
            qweight = layer.qweight
            result = []
            flat_map = getattr(layer.qweight, "flat_shard_map", None)
            for idx in shard_id:
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                if flat_map is not None:
                    off, rows, width = flat_map[idx]
                    w = qweight[off : off + rows * width].view(rows, width)
                else:
                    start, end, offset = layer.qweight.shard_offset_map[idx]
                    w = qweight[start:end, :offset].contiguous()
                result.append(fused_mul_mat_gguf(x, w, qweight_type))
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            try:
                out = fused_mul_mat_gguf(x, qweight, qweight_type)
            except RuntimeError as e:
                uninit = isinstance(qweight, UninitializedParameter)
                raise RuntimeError(
                    f"gguf apply failed in layer {getattr(layer, 'prefix', '?')} "
                    f"(qweight_type={qweight_type}, qweight={'UNINITIALIZED' if uninit else tuple(qweight.shape)}, "
                    f"x={tuple(x.shape)}): {e}"
                ) from e
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config
        self.fused_experts = None  # apply() asserts on it; upstream never sets it

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        # gate up proj
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        # gate down proj
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )

        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # GGUFMoEMethod: materialise the expert shards FusedMoE staged at load
        if hasattr(layer, "materialize_gguf_weights"):
            layer.materialize_gguf_weights()
            for _, p in layer.named_parameters():
                if getattr(p, "is_gguf_weight", False):
                    getattr(p, "data_container", []).clear()
                    getattr(p, "expert_data_map", {}).clear()
            _gguf_trim_host("after MoE materialisation")
        if _MOE_FP8:
            self._moe_fp8_prepare(layer)

    def _moe_fp8_prepare(self, layer: torch.nn.Module):
        """Repack raw Q6_K experts to 224-byte blocks and set one fp8 scale per tensor."""
        for name in ("w13", "w2"):
            wt = getattr(layer, f"{name}_qweight_type")
            if wt.weight_type == WeightType.Q6_K:
                w = getattr(layer, f"{name}_qweight")
                E, N, B = w.shape
                padded = Parameter(_repack_q6_k_padded(w.data.view(E * N, B)).view(E, N, -1),
                                   requires_grad=False)
                set_weight_attrs(padded, {k: v for k, v in vars(w).items() if not k.startswith("_")})
                layer.register_parameter(f"{name}_qweight", padded)
                wt.weight_type = _GGUF_Q6_K_PAD
        q13, q2 = int(layer.w13_qweight_type.weight_type), int(layer.w2_qweight_type.weight_type)
        if q13 not in _gguf_moe_fp8.SUPPORTED or q2 not in _gguf_moe_fp8.SUPPORTED:
            return
        layer._moe_fp8_ws13 = _gguf_moe_fp8.weight_scale(layer.w13_qweight, q13)
        layer._moe_fp8_ws2 = _gguf_moe_fp8.weight_scale(layer.w2_qweight, q2)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        assert self.fused_experts is None

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert self.moe_runner_config.activation == "silu", (
            "Only SiLU activation is supported."
        )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        topk_weights, topk_ids, _ = topk_output
        if (
            _MOE_FP8
            and getattr(layer, "_moe_fp8_ws13", None) is not None
            and x.dim() == 2
            and x.shape[0] >= _MOE_FP8_MIN_TOKENS
            and x.dtype == torch.bfloat16
        ):
            output = _gguf_moe_fp8.fused_moe_fp8(
                x.contiguous(), layer.w13_qweight, layer.w2_qweight, topk_weights, topk_ids,
                int(layer.w13_qweight_type.weight_type), int(layer.w2_qweight_type.weight_type),
                layer._moe_fp8_ws13, layer._moe_fp8_ws2, moe_align_block_size,
            )
            return StandardCombineInput(hidden_states=output)
        output = fused_moe_gguf(
            x=x,
            w1=layer.w13_qweight,
            w2=layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            qweight_type=layer.w13_qweight_type.weight_type,
            qweight_type2=layer.w2_qweight_type.weight_type,
            activation=moe_runner_config.activation,
        )
        return StandardCombineInput(hidden_states=output)


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]

        return apply_gguf_embedding(
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: list[torch.Tensor]


# =============================================================================
# NPU-specific implementations for Ascend hardware
# =============================================================================
def ggml_dequantize_ascend(
    qweight: torch.Tensor,
    qweight_type: int,
    rows: int,
    cols: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize GGML quantized weights for NPU.

    Uses gguf library's reference implementation which supports all GGML formats
    and is guaranteed to be correct. The dequantization runs on CPU during model
    loading, then the dequantized weights are transferred to NPU for inference.
    """

    # Move to CPU for dequantization using gguf library
    qweight_cpu = qweight.cpu().numpy()

    # Use gguf library's dequantize (supports all GGML formats)
    dequant_np = gguf_dequantize(qweight_cpu, qweight_type)

    # Convert to torch and move to target device
    result = torch.from_numpy(dequant_np).to(dtype=dtype, device=qweight.device)
    result = result.reshape(rows, cols)

    return result


class GGUFLinearAscendMethod(LinearMethodBase):
    """Linear method for GGUF on Ascend NPU."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            raise ValueError(
                f"Unsupported GGUF quantization type {WeightType(qweight_type)} in layer."
            )
        self._create_padded_weight_param(layer)
        # Pre-dequantize weights for faster inference
        self._pre_dequantize_weights(layer)

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def _pre_dequantize_weights(self, layer: torch.nn.Module):
        """Pre-dequantize GGML weights to FP16 for faster inference.

        This eliminates runtime dequantization overhead at the cost of more memory.
        """
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type

        if qweight_type in UNQUANTIZED_TYPES and qweight.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            layer.dequantized_weight = qweight
            return

        shard_id = getattr(qweight, "shard_id", None)
        has_shard_offset = hasattr(qweight, "shard_offset_map")

        if shard_id and has_shard_offset:
            # Handle sharded weights (QKV merged)
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            dequant_shards = []
            for idx in shard_id:
                start, end, offset = qweight.shard_offset_map[idx]
                shard_qtype = layer.qweight_type.shard_weight_type[idx]
                shard_data = qweight[start:end, :offset].contiguous()

                block_size, type_size = gguf.GGML_QUANT_SIZES[shard_qtype]
                shape = (
                    shard_data.shape[0],
                    shard_data.shape[1] // type_size * block_size,
                )
                dequant = ggml_dequantize_ascend(
                    shard_data, shard_qtype, *shape, self.params_dtype
                )
                dequant_shards.append(dequant)

            dequant_weight = torch.cat(dequant_shards, dim=0)
        else:
            # Handle single weight
            block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
            shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
            dequant_weight = ggml_dequantize_ascend(
                qweight, qweight_type, *shape, self.params_dtype
            )

        layer.dequantized_weight = dequant_weight

        if hasattr(layer, "qweight"):
            del layer.qweight
        if hasattr(layer, "qweight_type"):
            del layer.qweight_type

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Use pre-dequantized weight (always available after process_weights_after_loading)
        weight = layer.dequantized_weight
        out = x @ weight.T
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEAscendMethod(FusedMoEMethodBase):
    """MoE method for GGUF on Ascend NPU."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config
        self.w13_kernel = NPUUnquantMoEMethod()
        self.w2_kernel = NPUUnquantMoEMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

        # Store params_dtype for pre-dequantization
        self.params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Pre-dequantize MoE weights to FP16 for faster inference."""

        if hasattr(layer, "materialize_gguf_weights"):
            layer.materialize_gguf_weights()

        # Check if weights are actually loaded (not still UninitializedParameter/empty)
        w13_qweight = layer.w13_qweight
        w13_qtype = layer.w13_qweight_type.weight_type

        # Pre-dequantize w13 weights (gate+up projections)
        if w13_qtype not in UNQUANTIZED_TYPES:
            num_experts = w13_qweight.shape[0]
            w13_dequant_list = []

            block_size, type_size = gguf.GGML_QUANT_SIZES[w13_qtype]

            for e in range(num_experts):
                qweight_cpu = w13_qweight[e].cpu().numpy()
                rows = w13_qweight[e].shape[0]
                cols = w13_qweight[e].shape[1] // type_size * block_size

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w13_qtype)
                dequant = (
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w13_qweight.device)
                    .reshape(rows, cols)
                    .contiguous()
                )
                w13_dequant_list.append(dequant)

            w13_full = torch.stack(w13_dequant_list, dim=0)
            layer.register_buffer(
                "w13_dequant", npu_format_cast(w13_full), persistent=False
            )
        else:
            layer.register_buffer(
                "w13_dequant", npu_format_cast(w13_qweight.data), persistent=False
            )

        # Pre-dequantize w2 weights (down projection)
        w2_qweight = layer.w2_qweight
        w2_qtype = layer.w2_qweight_type.weight_type

        if w2_qtype not in UNQUANTIZED_TYPES:
            num_experts = w2_qweight.shape[0]
            w2_dequant_list = []

            block_size, type_size = gguf.GGML_QUANT_SIZES[w2_qtype]

            for e in range(num_experts):
                qweight_cpu = w2_qweight[e].cpu().numpy()
                rows = w2_qweight[e].shape[0]
                cols = w2_qweight[e].shape[1] // type_size * block_size

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w2_qtype)
                dequant = (
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w2_qweight.device)
                    .reshape(rows, cols)
                    .contiguous()
                )
                w2_dequant_list.append(dequant)

            w2_full = torch.stack(w2_dequant_list, dim=0)

            layer.register_buffer(
                "w2_dequant", npu_format_cast(w2_full), persistent=False
            )
        else:
            layer.register_buffer(
                "w2_dequant", npu_format_cast(w2_qweight.data), persistent=False
            )

        if hasattr(layer, "w2_qweight"):
            del layer.w2_qweight
        if hasattr(layer, "w13_qweight"):
            del layer.w13_qweight

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"quant_type": "gguf"})

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        layer.w13_kernel = self.w13_kernel
        layer.w2_kernel = self.w2_kernel
        moe_runner_config.layer = layer
        moe_runner_config.use_tp_all_gather_activation = True
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.ASCEND
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        quant_info = AscendQuantInfo(
            w13_weight=layer.w13_dequant,
            w2_weight=layer.w2_dequant,
            w13_weight_bias=getattr(layer, "w13_weight_bias", None),
            w2_weight_bias=getattr(layer, "w2_weight_bias", None),
            w13_scale_bias=getattr(layer, "w13_scale_bias", None),
            w2_scale_bias=getattr(layer, "w2_scale_bias", None),
        )
        return self.runner.run(dispatch_output, quant_info)


class GGUFEmbeddingAscendMethod(GGUFLinearAscendMethod):
    """Embedding method for GGUF on Ascend NPU."""

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.embedding(layer.dequantized_weight, x)
