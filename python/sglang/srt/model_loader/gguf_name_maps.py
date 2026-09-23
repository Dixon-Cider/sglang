# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Per-architecture GGUF -> HF tensor name maps.

``GGUFModelLoader`` normally derives this map from ``gguf.get_tensor_name_map``,
which only covers architectures upstream gguf-py knows, and from a meta-device
``AutoModelForCausalLM.from_config`` to enumerate the HF parameter names. Neither
works for an architecture that lives outside transformers, so those are supplied
here instead.

A builder returns the complete ``{gguf_tensor_name: hf_param_name}`` map. Any
GGUF tensor left out of the map is skipped by ``gguf_quant_weights_iterator``,
which is how dummy tensors are dropped.
"""

from typing import Callable, Dict

from transformers import PretrainedConfig

# Sandwich naming: ffn_norm is the pre-FFN norm.
_MUSE_GLIMMER_LAYER_TENSORS = {
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attn_norm",
    "ffn_norm": "post_attention_layernorm",
    "post_ffw_norm": "post_ffn_norm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_gate": "self_attn.output_gate_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

_MUSE_GLIMMER_GLOBAL_TENSORS = {
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output": "lm_head",
}

# attn_q_norm/attn_k_norm omitted: Muse Glimmer's QK-norm is non-parametric.


def build_muse_glimmer_name_map(config: PretrainedConfig) -> Dict[str, str]:
    name_map = {
        f"{gguf}.weight": f"{hf}.weight"
        for gguf, hf in _MUSE_GLIMMER_GLOBAL_TENSORS.items()
    }
    for layer in range(config.num_hidden_layers):
        for gguf, hf in _MUSE_GLIMMER_LAYER_TENSORS.items():
            name_map[f"blk.{layer}.{gguf}.weight"] = f"model.layers.{layer}.{hf}.weight"
    return name_map


# Keyed by HF ``config.model_type`` (loader.py looks it up with that), which is
# not the GGUF ``general.architecture`` that GGUF_NATIVE_CONFIG_BUILDERS uses:
# llama.cpp spells the arch "muse-glimmer" while the HF config says "muse_glimmer".
# --------------------------------------------------------------------------
# Qwen3.5 / 3.6 / 3.8 (llama.cpp arch "qwen35"): hybrid GatedDeltaNet + attention,
# VL top-level config, optional MTP block appended as blk.{num_hidden_layers + i}.
# gguf-py's tensor map covers the text tensors but not A_log / dt_bias (no
# ".weight" suffix) nor the nextn.* MTP tensors, and the loader's meta-model
# enumeration cannot see the MTP module, so the whole map is spelled out here.
_QWEN3_5_ATTN_TENSORS = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
}
_QWEN3_5_GDN_TENSORS = {
    "attn_qkv": "linear_attn.in_proj_qkv",
    "attn_gate": "linear_attn.in_proj_z",
    "ssm_beta": "linear_attn.in_proj_b",
    "ssm_alpha": "linear_attn.in_proj_a",
    "ssm_out": "linear_attn.out_proj",
    "ssm_norm": "linear_attn.norm",
    "ssm_conv1d": "linear_attn.conv1d",
}
_QWEN3_5_COMMON_TENSORS = {
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
_QWEN3_5_MOE_TENSORS = {
    "ffn_gate_inp": "mlp.gate",
    "ffn_gate_inp_shexp": "mlp.shared_expert_gate",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
    "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj",
}
_QWEN3_5_NEXTN_TENSORS = {
    "nextn.eh_proj": "fc",
    "nextn.enorm": "pre_fc_norm_embedding",
    "nextn.hnorm": "pre_fc_norm_hidden",
    "nextn.shared_head_norm": "norm",
}


def _qwen3_5_text_config(config: PretrainedConfig) -> PretrainedConfig:
    return getattr(config, "text_config", None) or config


def build_qwen3_5_name_map(config: PretrainedConfig) -> Dict[str, str]:
    tc = _qwen3_5_text_config(config)
    n = tc.num_hidden_layers
    layer_types = getattr(tc, "layer_types", None) or [
        ("full_attention" if (i + 1) % getattr(tc, "full_attention_interval", 4) == 0 else "linear_attention")
        for i in range(n)
    ]
    name_map = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    # SGLang normalises the dense text config with MoE defaults (num_experts=512),
    # so decide by model_type, not by the field.
    is_moe = "moe" in (str(getattr(config, "model_type", "")) + str(getattr(tc, "model_type", "")))
    for i, lt in enumerate(layer_types):
        hf = f"model.layers.{i}"
        for g, h in _QWEN3_5_COMMON_TENSORS.items():
            if is_moe and g.startswith("ffn_"):
                continue  # MoE layers: routed experts come from the iterator's ffn_*_exps path
            name_map[f"blk.{i}.{g}.weight"] = f"{hf}.{h}.weight"
        if is_moe:
            for g, h in _QWEN3_5_MOE_TENSORS.items():
                name_map[f"blk.{i}.{g}.weight"] = f"{hf}.{h}.weight"
        if lt == "full_attention":
            for g, h in _QWEN3_5_ATTN_TENSORS.items():
                name_map[f"blk.{i}.{g}.weight"] = f"{hf}.{h}.weight"
        else:
            for g, h in _QWEN3_5_GDN_TENSORS.items():
                name_map[f"blk.{i}.{g}.weight"] = f"{hf}.{h}.weight"
            name_map[f"blk.{i}.ssm_a"] = f"{hf}.linear_attn.A_log"
            name_map[f"blk.{i}.ssm_dt.bias"] = f"{hf}.linear_attn.dt_bias"
    n_mtp = getattr(tc, "mtp_num_hidden_layers", None) or getattr(tc, "num_nextn_predict_layers", 0) or 0
    for j in range(n_mtp):
        b = n + j
        hf = f"mtp.layers.{j}"
        for g, h in {**_QWEN3_5_COMMON_TENSORS, **_QWEN3_5_ATTN_TENSORS}.items():
            name_map[f"blk.{b}.{g}.weight"] = f"{hf}.{h}.weight"
        for g, h in _QWEN3_5_NEXTN_TENSORS.items():
            name_map[f"blk.{b}.{g}.weight"] = f"mtp.{h}.weight"
    return name_map


# HF-side names whose GGUF tensor is quantized but whose SGLang parameter is a
# plain nn.Linear (no qweight): dequantize on load instead.
_QWEN3_5_DEQUANT_ON_LOAD = ("mtp.fc.weight",)


def qwen3_5_gguf_weights(gguf_file: str, name_map: Dict[str, str], weights):
    """Wrap gguf_quant_weights_iterator output with the qwen35 value conventions.

    llama.cpp's converter stores every RMSNorm weight as (w + 1) except the GDN
    gated norm, ssm_a as -exp(A_log), and conv1d squeezed to 2-D. SGLang's
    GemmaRMSNorm adds the 1 itself, wants A_log, and keeps conv1d 3-D.
    """
    import gguf
    import torch

    reader = gguf.GGUFReader(gguf_file)
    qtype = {name_map[t.name]: t.tensor_type for t in reader.tensors if t.name in name_map}
    dequant_q = {n.replace("weight", "qweight") for n in _QWEN3_5_DEQUANT_ON_LOAD}
    dequant_t = {n.replace("weight", "qweight_type") for n in _QWEN3_5_DEQUANT_ON_LOAD}
    for name, tensor in weights:
        if name in dequant_t:
            continue
        if name in dequant_q:
            hf = name.replace("qweight", "weight")
            f32 = gguf.quants.dequantize(tensor.cpu().numpy(), qtype[hf])
            yield hf, torch.from_numpy(f32)
            continue
        if name.endswith("linear_attn.A_log"):
            tensor = torch.log(-tensor.float())
        elif name.endswith("linear_attn.conv1d.weight") and tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        elif name.endswith("shared_expert_gate.weight") and tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # GGUF stores the MoE shared-expert gate as [hidden]; SGLang wants [1, hidden]
        elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            tensor = tensor.float() - 1.0
        elif name.endswith(("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")):
            tensor = tensor.float() - 1.0
        yield name, tensor


GGUF_HF_NAME_MAP_BUILDERS: Dict[str, Callable[[PretrainedConfig], Dict[str, str]]] = {
    "muse_glimmer": build_muse_glimmer_name_map,
    "qwen3_5": build_qwen3_5_name_map,
    "qwen3_5_text": build_qwen3_5_name_map,
    "qwen3_5_moe": build_qwen3_5_name_map,
    "qwen3_5_moe_text": build_qwen3_5_name_map,
}

# Keyed like GGUF_HF_NAME_MAP_BUILDERS; wraps the loader's weight iterator.
GGUF_HF_WEIGHT_TRANSFORMS: Dict[str, Callable] = {
    "qwen3_5": qwen3_5_gguf_weights,
    "qwen3_5_text": qwen3_5_gguf_weights,
    "qwen3_5_moe": qwen3_5_gguf_weights,
    "qwen3_5_moe_text": qwen3_5_gguf_weights,
}
