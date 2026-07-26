"""Model presets for the transport program.

Slimmed at bootstrap (2026-07-26) from the anamnesis pipeline's `config.py`:
only `ModelPreset`/`MODEL_PRESETS` travel — the extraction/signature machinery
(ExtractionConfig, run registry, calibration paths) stays behind. Fields kept
are the ones the transport scripts and probes actually consume (`torch_dtype`
above all) plus the architecture facts useful for site-grid work. Roster
growth for P2 adds entries here; every addition should carry the same
verified-against-config.json discipline the dsv2-lite entry documents.
"""
from __future__ import annotations

from pydantic import BaseModel


class ModelPreset(BaseModel):
    """Per-model architecture + decode facts.

    Single source of truth for what varies between roster models. `torch_dtype`
    is load-bearing (probes forward in it); the rest are verified architecture
    reference (layer/dim/head counts, EOS ids, native temperature).
    """

    model_id: str
    torch_dtype: str
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    temperature: float
    eos_token_ids: list[int]
    # Per-layer attention type for interleaved-attention architectures
    # (Gemma-3 class: 5 local sliding-window : 1 global). None = all-global
    # (Llama/Qwen/OLMo full-context attention at every layer). Cross-model
    # comparisons should prefer GLOBAL layers on interleaved architectures.
    attention_layer_types: dict[int, str] | None = None


MODEL_PRESETS: dict[str, ModelPreset] = {
    "8b": ModelPreset(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype="bfloat16",
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        temperature=0.6,
        eos_token_ids=[128001, 128008, 128009],
    ),
    "3b": ModelPreset(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype="float16",
        num_layers=28,
        hidden_dim=3072,
        num_attention_heads=24,
        num_kv_heads=8,
        head_dim=128,
        temperature=0.7,
        eos_token_ids=[128001, 128009],
    ),
    "olmo2-7b": ModelPreset(
        # BASE model — no chat template (bare prompts only; raw arm only, per
        # the arm-consistency rule add-7.1). Full MHA (num_kv_heads == heads).
        model_id="allenai/OLMo-2-1124-7B",
        torch_dtype="bfloat16",
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
        temperature=0.7,
        eos_token_ids=[100257],
    ),
    "gemma3-27b": ModelPreset(
        # Gemma3ForConditionalGeneration (multimodal wrapper) — decoder layers
        # resolve via extraction.hooks.decoder_layers(). 5:1 local:global
        # attention interleave (sliding_window=1024; global at (i+1) % 6 == 0).
        # Native sampling per the model card: temperature 1.0, top_p 0.95.
        model_id="google/gemma-3-27b-it",
        torch_dtype="bfloat16",
        num_layers=62,
        hidden_dim=5376,
        num_attention_heads=32,
        num_kv_heads=16,
        head_dim=128,
        temperature=1.0,
        eos_token_ids=[1, 106],
        attention_layer_types={0: "local", 11: "global", 23: "global",
                               35: "global", 41: "global", 53: "global",
                               59: "global"},
    ),
    "qwen-7b": ModelPreset(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        torch_dtype="bfloat16",
        num_layers=28,
        hidden_dim=3584,
        num_attention_heads=28,
        num_kv_heads=4,
        head_dim=128,
        temperature=0.7,
        eos_token_ids=[151643, 151645],
    ),
    "dsv2-lite": ModelPreset(
        # DeepSeek-V2-Lite-Chat — MoE (2 shared + 64 routed experts, top-6
        # greedy; first_k_dense_replace=1, so layer 0 is a dense MLP and layers
        # 1–26 are MoE). Load via the NATIVE transformers `deepseek_v2`
        # integration with trust_remote_code=False — the bundled auto_map
        # remote code has a DIFFERENT internal structure. MLA attention: no
        # k_proj/v_proj module; head_dim below is v_head_dim (qk_head_dim=192).
        # All numbers verified against the downloaded config.json (2026-07-17).
        model_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
        torch_dtype="bfloat16",
        num_layers=27,
        hidden_dim=2048,
        num_attention_heads=16,
        num_kv_heads=16,
        head_dim=128,
        temperature=0.3,           # generation_config.json = 0.3 (native top_p 0.95)
        eos_token_ids=[100001],
    ),
}
