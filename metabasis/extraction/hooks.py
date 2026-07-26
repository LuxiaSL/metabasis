"""Minimal model-hook machinery for the transport program.

Vendored verbatim from the anamnesis extraction stack (`model_loader.py`,
public `anamnesis-pl` repo) at metabasis bootstrap, 2026-07-26 — the ONLY two
pieces of that 937-line module the transport program needs:

  * `decoder_layers()` — architecture-resolution helper (Llama/Qwen/OLMo-2
    direct, Gemma-3 multimodal wrapper via `language_model`).
  * `ResidualWriteSpec` / `attach_residual_write` — one residual-stream
    injection hook, used by the behavioral-write probes
    (`vmb_c3_entropy_replay`, the owl/needle probes).

Nothing else travels: no k_proj/gate_proj pre-RoPE capture, no HookState, no
LoadedModel, no eager-attention plumbing (charter §2f: "none of the anamnesis
extraction stack"). The slim state collector hand-rolls its own read-side
capture hook and imports only `decoder_layers` from here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

logger = logging.getLogger("metabasis.hooks")


def decoder_layers(model):
    """Decoder layer list across architectures.

    Llama/Qwen/OLMo-2: `model.model.layers`. Gemma-3 ships as
    Gemma3ForConditionalGeneration — a multimodal wrapper whose text decoder
    nests under `language_model`; hook paths must resolve through it.
    """
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        return inner.layers
    lm = getattr(inner, "language_model", None) or getattr(model, "language_model", None)
    if lm is not None:
        lm_inner = getattr(lm, "model", lm)
        if hasattr(lm_inner, "layers"):
            return lm_inner.layers
    raise AttributeError(
        f"cannot locate decoder layers on {type(model).__name__} — extend "
        "decoder_layers() for this architecture"
    )


# ── Residual-stream write (steering injection) ────────────────────────────────
# forward_pre_hook on a decoder layer adds alpha * unit(vector) to the residual
# stream entering that layer, at chosen sequence positions. Used for steering
# during replay of a fixed continuation (matched-token channel) and free-gen
# dose ladders. alpha=0 must reproduce the unperturbed forward exactly.


@dataclass
class ResidualWriteSpec:
    """One residual-stream injection: add `alpha * vector/||vector||` to the
    hidden states entering decoder layer `layer_idx`.

    start_pos/end_pos select ABSOLUTE sequence positions (end exclusive;
    None = unbounded). For steering over generated tokens only, set
    start_pos = prompt_length. Positional gating reads the layer's
    `cache_position` kwarg (absolute positions, present on every HF call path
    in transformers 5.x), so ONE spec is valid under single-forward replay AND
    incremental generate() (per-step seq_len=1 with KV cache) — no per-step
    toggling. If cache_position is absent the hook falls back to local-index
    slicing (correct only for full-sequence forwards) and RAISES on an
    ambiguous incremental step rather than silently mis-injecting.

    start_pos is read at every hook call, so a caller may mutate it per
    generation (prompt lengths vary) on a hook registered once.
    """

    layer_idx: int
    vector: Tensor              # [hidden_dim]; normalized at registration
    alpha: float
    start_pos: int | None = None
    end_pos: int | None = None
    normalize: bool = True


def _make_residual_write_pre_hook(
    spec: ResidualWriteSpec,
    enabled_flag: dict[str, bool],
    stats: dict[str, Any] | None = None,
) -> Any:
    """forward_pre_hook (with_kwargs) injecting spec into the layer's input hidden states.

    The decoder layer receives hidden_states as args[0] (HF Llama-class layers).
    Returns modified (args, kwargs). Positional gating uses the `cache_position`
    kwarg (absolute positions) when present — valid under prefill, incremental
    seq_len=1 steps, and single-forward replay alike. Injection tensor is
    cast/moved lazily to the input's dtype/device once, then cached on the spec
    closure (cache is keyed by alpha too, so per-cell alpha mutation is safe).
    `stats` (optional) accumulates calls/positions-injected/saw-cache-position
    for smoke assertions.
    """
    cache: dict[str, Tensor] = {}

    def pre_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not enabled_flag["enabled"] or spec.alpha == 0.0:
            return args, kwargs
        cache_position = kwargs.get("cache_position")
        if not args or not isinstance(args[0], Tensor):
            # Defensive: some architectures pass hidden_states as a kwarg.
            hs = kwargs.get("hidden_states")
            if hs is None:
                logger.warning("residual write hook: no hidden_states found; skipping injection")
                return args, kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = _inject(hs, cache_position)
            return args, kwargs
        new_args = (_inject(args[0], cache_position),) + tuple(args[1:])
        return new_args, kwargs

    def _inject(hs: Tensor, cache_position: Tensor | None) -> Tensor:
        # id(spec.vector) in the key: callers may swap the vector tensor on a live
        # spec (pilot gates iterate vectors); a swapped tensor must never reuse a
        # stale cached delta.
        key = f"{hs.device}_{hs.dtype}_{spec.alpha}_{id(spec.vector)}"
        if key not in cache:
            v = spec.vector.detach().to(device=hs.device, dtype=torch.float32)
            if spec.normalize:
                v = v / v.norm().clamp_min(1e-12)
            cache[key] = (spec.alpha * v).to(dtype=hs.dtype)
        delta = cache[key]
        bounded = spec.start_pos is not None or spec.end_pos is not None
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
            stats["saw_cache_position"] = stats.get("saw_cache_position", False) or (
                cache_position is not None
            )
        if cache_position is not None:
            # Absolute-position gating: one semantics for prefill, incremental
            # decoding, and single-forward replay.
            s = spec.start_pos if spec.start_pos is not None else 0
            e = spec.end_pos if spec.end_pos is not None else int(cache_position.max().item()) + 1
            # mask must live on the hidden-states device — cache_position can be on a different
            # device (multi-GPU / CPU cache_position), which crashes out[:, mask, :] with
            # "indices should be ... on the same device as the indexed tensor".
            mask = ((cache_position >= s) & (cache_position < e)).to(hs.device)  # [seq_len] bool
            n_inj = int(mask.sum().item())
            if stats is not None:
                stats["positions"] = stats.get("positions", 0) + n_inj
            if n_inj == 0:
                return hs
            if n_inj == hs.shape[1]:
                return hs + delta
            out = hs.clone()
            out[:, mask, :] = out[:, mask, :] + delta
            return out
        # Fallback: local-index slicing — correct ONLY for full-sequence forwards.
        if bounded and hs.shape[1] == 1:
            raise RuntimeError(
                "residual write hook: positionally-bounded spec on a seq_len=1 "
                "forward without cache_position — cannot determine the absolute "
                "position; refusing to silently mis-inject (incremental decoding "
                "requires the cache_position kwarg)"
            )
        s = spec.start_pos if spec.start_pos is not None else 0
        e = spec.end_pos if spec.end_pos is not None else hs.shape[1]
        s = max(0, min(s, hs.shape[1]))
        e = max(s, min(e, hs.shape[1]))
        if stats is not None:
            stats["positions"] = stats.get("positions", 0) + (e - s)
        if s == 0 and e == hs.shape[1]:
            return hs + delta
        out = hs.clone()
        out[:, s:e, :] = out[:, s:e, :] + delta
        return out

    return pre_hook


@dataclass
class ResidualWriteHandle:
    """Removable handle for a registered residual write; also supports toggling.

    `stats` accumulates {calls, positions, saw_cache_position} across forwards
    (reset via reset_stats) — smoke tests assert injected-position counts against
    expectation. `spec.start_pos`/`spec.alpha` may be mutated between generations
    on a live handle (read per hook call)."""

    spec: ResidualWriteSpec
    _handle: Any
    _enabled_flag: dict[str, bool]
    stats: dict[str, Any] = field(default_factory=dict)

    def disable(self) -> None:
        self._enabled_flag["enabled"] = False

    def enable(self) -> None:
        self._enabled_flag["enabled"] = True

    def remove(self) -> None:
        self._handle.remove()

    def reset_stats(self) -> None:
        self.stats.clear()


def attach_residual_write(model: Any, spec: ResidualWriteSpec) -> ResidualWriteHandle:
    """Register a residual-write injection on a BARE HF model.

    Validates layer_idx/hidden_dim against model.config. Shared by every
    generation-side probe in the program.
    """
    layers = decoder_layers(model)
    n_layers = len(layers)
    # wrapper-aware: Gemma3ForConditionalGeneration's Gemma3Config nests dims under
    # text_config (no top-level hidden_size); Llama/Qwen/OLMo expose it directly.
    _cfg = model.config
    hidden_dim = int(getattr(_cfg, "hidden_size", None)
                     or getattr(getattr(_cfg, "text_config", None), "hidden_size"))
    if not 0 <= spec.layer_idx < n_layers:
        raise ValueError(
            f"residual write layer_idx {spec.layer_idx} out of range (model has {n_layers} layers)"
        )
    if spec.vector.numel() != hidden_dim:
        raise ValueError(
            f"residual write vector has {spec.vector.numel()} elements, expected hidden_dim={hidden_dim}"
        )
    enabled_flag = {"enabled": True}
    stats: dict[str, Any] = {}
    handle = layers[spec.layer_idx].register_forward_pre_hook(
        _make_residual_write_pre_hook(spec, enabled_flag, stats), with_kwargs=True
    )
    logger.info(
        f"Residual write registered: layer {spec.layer_idx}, alpha={spec.alpha}, "
        f"pos=[{spec.start_pos},{spec.end_pos})"
    )
    return ResidualWriteHandle(spec=spec, _handle=handle, _enabled_flag=enabled_flag, stats=stats)
