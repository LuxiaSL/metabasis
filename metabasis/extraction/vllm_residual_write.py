"""The metabasis residual write for a vLLM-served model — OUR contract, ours only.

UNSTAMPED (C§8). NO SCIENCE CELL FIRES ON THIS LANE. The prod-lane pre-statement,
with distribution-level tolerances, goes to Luxia before any bridge or science run.

WHAT THIS IS. The vLLM-side sibling of `extraction/hooks.py`'s
`attach_residual_write`. Same intervention, same arithmetic, different serving
substrate. It is written from OUR contract; the third-party module on the cluster
(`assistant-axis-exp/scripts/vllm_steering.py`, read-only reference, never forked and
never executed) informed exactly three INTEGRATION points and nothing semantic:

  (i)   a `torch.library.custom_op` is the way to keep a mutating step out of
        dynamo's trace and act as a piecewise CUDA-graph splitting point;
  (ii)  per-request configuration reaches TP workers through a filesystem/shm
        channel that every rank reads, so all ranks apply the same modification and
        the residual streams stay consistent;
  (iii) a decoder layer can be replaced in the model's layer list by a wrapper
        module of the same call signature.

EVERYTHING SEMANTIC IS OURS AND DIFFERS FROM THEIRS ON FOUR AXES. Those four deltas
ARE the specification of this file:

  1. SITE — WE WRITE THE LAYER *INPUT*, THEY WRITE THE LAYER *OUTPUT*.
     `hooks.py` registers a `register_forward_pre_hook` on `decoder_layers(model)[s]`,
     so the tensor written is the residual stream ENTERING layer `s` -- and that is
     the very same tensor §2.5 takes its per-token median norm over, which is what
     makes the dose "resolved against the quantity it perturbs". This file wraps
     layer `s` and writes its INPUT before delegating. NO OFF-BY-ONE IS INTRODUCED
     and none is needed.

         our site s  ==  input of layer s  ==  output of layer s-1
         an output-writing convention would have to wrap layer s-1 to mean the
         same thing; ours wraps layer s and writes the input, which is the direct
         translation of the pre-hook and needs no arithmetic at all.

     THE MAPPING IS PROVEN BY VALUE, NEVER BY INDEX ARITHMETIC:
     `capture_input_hidden_states()` records exactly the tensor this wrapper is
     about to write, and the smoke compares it against the tensor the HF pre-hook
     sees on a matched forward. Equality by value is the proof.

  2. DOSE -- NORM-RELATIVE AND MEASURED IN-JOB, NOT AN ABSOLUTE SCALAR.
     `alpha = alpha_frac x measured_per_token_median_resid_norm` (§2.5). This file
     never invents a strength: it takes an ALREADY-RESOLVED absolute `alpha`, and
     `measure_per_token_median_resid_norm_from_captures()` reproduces the engine's
     own definition (per-token L2 norms of the layer-input hidden states over
     non-padding positions, median) from captures taken through THIS wrapper, so the
     dose is resolved against the same quantity on both lanes.

  3. VECTOR -- BANKED UNIT, NORM MATCHED AT ATTACH. The delta is computed exactly as
     `hooks.py` computes it: normalise in float32, scale by alpha in float32, then
     cast once to the hidden-states dtype, and cache it. Bit-for-bit the same
     recipe, so a difference between lanes can never be the delta's arithmetic.

  4. SPAN -- GENERATED POSITIONS ONLY. `hooks.py` gates on absolute `cache_position`
     in `[start_pos, end_pos)` with `start_pos = prompt_length`. Here the same rule
     is expressed per request: a token is injected iff its absolute position is >=
     that request's prompt length. Prefill tokens are NEVER injected. vLLM makes
     this cleaner than HF did, not looser.

AND TWO THINGS THIS FILE REFUSES TO HAVE:

  * NO RENORM. Rescaling the modified state back to its original norm is a
    DIFFERENT operation, not a different implementation. There is no flag.
  * NO floor/ceil (activation capping). Those are lesion-adjacent; the engine
    refuses lesion recipes at construction and so does this lane. There is no
    intervention-type field at all -- the only operation is `h += alpha * v_hat`.

`alpha == 0.0` short-circuits to a bitwise no-op, exactly as §2.4 requires of the
baseline cell.

IMPORT DISCIPLINE: vLLM is imported LAZILY, inside the functions that need it, so
this module imports, type-checks and SELFTESTS on a machine with no vLLM at all --
which is where its unit math is proven.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger("metabasis.vllm_residual_write")

#: Where per-request steering configuration is published so that EVERY tensor-parallel
#: rank reads the same bytes (integration point (ii)). Under the node data root by the
#: standing placement rule; /dev/shm is used only when explicitly requested.
DEFAULT_CONFIG_DIR = Path(
    os.environ.get("METABASIS_STEERING_DIR",
                   "/models/metabasis-behavioral/vllm-lane/steering"))


class SteeringContractViolation(RuntimeError):
    """A named refusal. The lane never degrades quietly to a different operation."""


@dataclass(frozen=True)
class MetabasisSteeringSpec:
    """One residual-stream write, to the metabasis contract and nothing else.

    Deliberately NOT parameterised by intervention type, renormalisation, or an
    absolute strength: those are the three doors through which another lane's
    semantics would enter. The only operation is `h += alpha * v/||v||` at the
    INPUT of decoder layer `layer_idx`, over generated positions only.
    """

    layer_idx: int
    vector: Tensor                      # [hidden]; banked unit, re-normalised here
    alpha: float                        # ALREADY RESOLVED: alpha_frac x §2.5 norm
    alpha_frac: Optional[float] = None  # recorded for provenance, never used in math
    measured_norm: Optional[float] = None
    normalize: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.layer_idx, int) or self.layer_idx < 0:
            raise SteeringContractViolation(
                f"layer_idx must be a non-negative int, got {self.layer_idx!r}")
        if not isinstance(self.vector, Tensor) or self.vector.dim() != 1:
            raise SteeringContractViolation(
                "vector must be a 1-D tensor of shape [hidden]")
        if self.vector.numel() == 0:
            raise SteeringContractViolation("vector is empty")
        if not np.isfinite(float(self.alpha)):
            raise SteeringContractViolation(f"alpha {self.alpha!r} is not finite")
        if (self.alpha_frac is not None and self.measured_norm is not None
                and self.alpha != 0.0):
            want = float(self.alpha_frac) * float(self.measured_norm)
            if not np.isclose(want, float(self.alpha), rtol=1e-9, atol=0.0):
                raise SteeringContractViolation(
                    f"alpha {self.alpha!r} is not alpha_frac x measured_norm "
                    f"({self.alpha_frac!r} x {self.measured_norm!r} = {want!r}); §2.5 "
                    "resolves the dose against the measured norm and this lane will "
                    "not carry a dose that was resolved some other way")

    def provenance(self) -> dict[str, Any]:
        return {"layer_idx": self.layer_idx, "alpha": float(self.alpha),
                "alpha_frac": self.alpha_frac, "measured_norm": self.measured_norm,
                "normalize": self.normalize,
                "vector_dim": int(self.vector.numel()),
                "operation": "h += alpha * v/||v|| at the INPUT of layer_idx",
                "span": "generated positions only (absolute pos >= prompt_len)",
                "renorm": False, "capping": False}


def steering_delta(vector: Tensor, alpha: float, *, dtype: torch.dtype,
                   device: torch.device, normalize: bool = True) -> Tensor:
    """The delta, computed EXACTLY as `hooks.py` computes it.

    float32 normalise -> float32 scale -> ONE cast to the hidden dtype. The order is
    load-bearing: normalising after the cast, or scaling in bf16, would put a
    different number into the residual stream than the byte-exact lane puts there.
    """
    v = vector.detach().to(device=device, dtype=torch.float32)
    if normalize:
        v = v / v.norm().clamp_min(1e-12)
    return (float(alpha) * v).to(dtype=dtype)


class MetabasisSteeringLayer(nn.Module):
    """Wraps decoder layer `layer_idx` and writes ITS INPUT (integration point (iii)).

    The wrapper is call-signature transparent: it forwards `*args, **kwargs` to the
    wrapped layer unchanged except for the hidden-states tensor it has written. It
    holds no opinion about vLLM's batching -- which tokens are injectable is decided
    by `position_mask`, supplied by the runner, so chunked prefill and continuous
    batching cannot silently change the span.
    """

    def __init__(self, wrapped_layer: nn.Module, layer_idx: int,
                 hidden_dim: int) -> None:
        super().__init__()
        self.wrapped_layer = wrapped_layer
        self.layer_idx = int(layer_idx)
        self.hidden_dim = int(hidden_dim)
        self._spec: Optional[MetabasisSteeringSpec] = None
        self._delta_cache: dict[str, Tensor] = {}
        self._position_mask: Optional[Tensor] = None
        self._capture: Optional[list[Tensor]] = None
        self.stats: dict[str, Any] = {"calls": 0, "tokens_injected": 0,
                                      "tokens_seen": 0, "prefill_tokens_skipped": 0}

    # -- configuration -------------------------------------------------------
    def set_spec(self, spec: Optional[MetabasisSteeringSpec]) -> None:
        if spec is not None and spec.vector.numel() != self.hidden_dim:
            raise SteeringContractViolation(
                f"vector has {spec.vector.numel()} elements, expected hidden_dim="
                f"{self.hidden_dim}")
        if spec is not None and spec.layer_idx != self.layer_idx:
            raise SteeringContractViolation(
                f"spec names layer {spec.layer_idx} but this wrapper is layer "
                f"{self.layer_idx} -- a mis-sited write is the one failure this "
                "lane exists to make impossible")
        self._spec = spec
        self._delta_cache.clear()

    def set_position_mask(self, mask: Optional[Tensor]) -> None:
        """Per-token injectability for the CURRENT forward.

        `mask[i]` is True iff flattened token i is a GENERATED position of its
        request (absolute position >= that request's prompt length). The runner owns
        this because only the runner knows the batch's request layout; the wrapper
        refuses to guess.
        """
        self._position_mask = mask

    def capture_input_hidden_states(self, enabled: bool) -> None:
        """Record the exact tensor this wrapper writes -- the site-mapping proof."""
        self._capture = [] if enabled else None

    def take_captures(self) -> list[Tensor]:
        out = list(self._capture or [])
        self._capture = []
        return out

    def reset_stats(self) -> None:
        self.stats = {"calls": 0, "tokens_injected": 0, "tokens_seen": 0,
                      "prefill_tokens_skipped": 0}

    # -- the write -----------------------------------------------------------
    def _write(self, hs: Tensor) -> Tensor:
        self.stats["calls"] += 1
        # HF hands us [batch, tokens, hidden]; vLLM hands us [tokens, hidden].
        # Both put the token axis at -2 only in the HF case, so read it explicitly.
        n_tok = int(hs.shape[0]) if hs.dim() == 2 else (
            int(hs.shape[-2]) if hs.dim() >= 3 else 0)
        self.stats["tokens_seen"] += n_tok
        if self._capture is not None:
            self._capture.append(hs.detach().clone())
        spec = self._spec
        if spec is None or spec.alpha == 0.0:
            # §2.4: the baseline runs WITH the wrapper attached and must be bitwise
            # identical to no wrapper at all.
            return hs
        key = f"{hs.device}_{hs.dtype}_{spec.alpha}_{id(spec.vector)}"
        delta = self._delta_cache.get(key)
        if delta is None:
            delta = steering_delta(spec.vector, spec.alpha, dtype=hs.dtype,
                                   device=hs.device, normalize=spec.normalize)
            self._delta_cache[key] = delta
        mask = self._position_mask
        if mask is None:
            raise SteeringContractViolation(
                "no position mask set for this forward. The span is "
                "generated-positions-only; injecting every token because the runner "
                "forgot to say which are generated would be a DIFFERENT experiment, "
                "so this lane refuses rather than assumes.")
        m = mask.to(device=hs.device, dtype=torch.bool)
        if m.numel() != n_tok:
            raise SteeringContractViolation(
                f"position mask has {m.numel()} entries for {n_tok} tokens")
        n_inj = int(m.sum().item())
        self.stats["tokens_injected"] += n_inj
        self.stats["prefill_tokens_skipped"] += (n_tok - n_inj)
        if n_inj == 0:
            return hs
        if n_inj == n_tok:
            return hs + delta
        out = hs.clone()
        if hs.dim() == 2:                       # vLLM: [tokens, hidden]
            out[m, :] = out[m, :] + delta
        else:                                   # HF: [batch, tokens, hidden]
            out[:, m, :] = out[:, m, :] + delta
        return out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Write THE RESIDUAL STREAM ENTERING the layer, then delegate.

        TWO CALL CONVENTIONS, AND THE DIFFERENCE IS NOT COSMETIC.

        HF (`transformers`) decoder layers take `hidden_states` FIRST and carry the
        whole residual stream in that one tensor -- which is what `hooks.py`'s
        pre-hook writes.

        vLLM decoder layers take `(positions, hidden_states, residual)` and use a
        FUSED add-and-norm: the layer begins with
            `hidden_states, residual = input_layernorm(hidden_states, residual)`
        whose first act is `residual = residual + hidden_states`. So in vLLM the
        residual stream entering layer k is **the SUM of two tensors**, not either
        one of them, and `hidden_states` alone is only a part of it. Writing
        `hidden_states` here would still land the delta in the right place (the sum
        is what the layernorm consumes), but writing the ACCUMULATOR is the exact
        analogue of the HF pre-hook and keeps the arithmetic in one tensor.

        `residual is None` marks the FIRST layer of the stack, where `hidden_states`
        alone IS the residual stream -- and that is the case our site s=0 would hit.

        This is a FIFTH delta from the reference module's world, and it is precisely
        the sort of thing index arithmetic cannot reveal: it was found by the
        wrapper refusing a malformed call, not by reading a layer index.
        """
        # -- vLLM fused-residual signature: (positions, hidden_states, residual) --
        if (len(args) >= 2 and isinstance(args[0], Tensor)
                and isinstance(args[1], Tensor) and args[0].dim() == 1):
            positions, hs = args[0], args[1]
            residual = args[2] if len(args) > 2 else kwargs.get("residual")
            if isinstance(residual, Tensor):
                new_res = self._write(residual)
                rest = tuple(args[3:]) if len(args) > 2 else ()
                if len(args) > 2:
                    return self.wrapped_layer(positions, hs, new_res, *rest, **kwargs)
                kwargs = dict(kwargs)
                kwargs["residual"] = new_res
                return self.wrapped_layer(positions, hs, **kwargs)
            # first layer of the stack: hidden_states IS the whole residual stream
            new_hs = self._write(hs)
            return self.wrapped_layer(positions, new_hs, *args[2:], **kwargs)

        # -- HF signature: hidden_states first ---------------------------------
        if args and isinstance(args[0], Tensor) and args[0].dim() >= 2:
            new_args = (self._write(args[0]),) + tuple(args[1:])
            return self.wrapped_layer(*new_args, **kwargs)
        hs = kwargs.get("hidden_states")
        if isinstance(hs, Tensor):
            kwargs = dict(kwargs)
            residual = kwargs.get("residual")
            if isinstance(residual, Tensor):
                kwargs["residual"] = self._write(residual)
            else:
                kwargs["hidden_states"] = self._write(hs)
            return self.wrapped_layer(*args, **kwargs)
        raise SteeringContractViolation(
            "could not locate the residual-stream tensor in the layer call; refusing "
            "to run an un-injected forward that the caller believes is steered")


# ── §2.5, reproduced from captures taken through THIS wrapper ─────────────────

def measure_per_token_median_resid_norm_from_captures(
        captures: list[Tensor], *, keep_mask: Optional[Tensor] = None) -> float:
    """The engine's §2.5 definition, applied to layer-INPUT captures.

    `run_behavioral_cells.HFNodeRuntime.measure_per_token_median_resid_norm` captures
    the hidden states entering `decoder_layers(model)[site]`, takes the L2 norm along
    the hidden axis, keeps non-padding positions, and returns the MEDIAN. This is
    that, over captures from the wrapper -- so both lanes measure the same quantity
    through the same definition and any difference is the serving stack, not the
    convention.
    """
    if not captures:
        raise SteeringContractViolation(
            "no captures to measure -- refusing to resolve a dose against nothing")
    vals: list[np.ndarray] = []
    for i, hs in enumerate(captures):
        h = hs.detach().float()
        norms = h.norm(dim=-1).reshape(-1).cpu().numpy()
        if keep_mask is not None:
            km = keep_mask.reshape(-1).cpu().numpy().astype(bool)
            if km.size != norms.size:
                raise SteeringContractViolation(
                    f"keep_mask has {km.size} entries for capture {i} with "
                    f"{norms.size} positions")
            norms = norms[km]
        vals.append(norms)
    allv = np.concatenate(vals) if len(vals) > 1 else vals[0]
    if allv.size == 0:
        raise SteeringContractViolation("every captured position was padding")
    return float(np.median(allv))


def resolve_alpha(alpha_frac: float, measured_norm: float) -> float:
    """§2.5's dose resolution, spelled out so the lane cannot drift from it."""
    if not np.isfinite(measured_norm) or measured_norm <= 0:
        raise SteeringContractViolation(
            f"§2.5 norm {measured_norm!r} is not a positive finite number")
    return float(alpha_frac) * float(measured_norm)


# ── span: generated positions only ────────────────────────────────────────────

def generated_position_mask(positions: Tensor, prompt_lens: Tensor) -> Tensor:
    """`hooks.py`'s `[start_pos, end_pos)` rule, per request, for a flat token batch.

    `positions[i]` is token i's ABSOLUTE position within its request; `prompt_lens[i]`
    is that request's prompt length. Injected iff `positions >= prompt_lens`, i.e.
    `start_pos = prompt_length`, `end_pos = None` -- the engine's own span.
    """
    if positions.shape != prompt_lens.shape:
        raise SteeringContractViolation(
            f"positions {tuple(positions.shape)} and prompt_lens "
            f"{tuple(prompt_lens.shape)} must align token-for-token")
    return positions >= prompt_lens


# ── per-request config plumbing (integration point (ii)) ──────────────────────

def write_steering_config(request_id: str, spec: Optional[MetabasisSteeringSpec], *,
                          config_dir: Path = DEFAULT_CONFIG_DIR,
                          vector_path: Optional[str] = None) -> Path:
    """Publish one request's steering so EVERY TP rank reads identical bytes.

    The VECTOR IS NOT INLINED -- the config names a banked `.npz` by path and sha, so
    a rank cannot be handed a different vector by a truncated write, and provenance
    stays the bank's rather than the message's.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "schema": "metabasis-vllm-steering/1",
        "unstamped": "C§8 -- prod lane, no science cell fires on it",
        "request_id": request_id,
        "spec": None if spec is None else spec.provenance(),
        "vector_path": vector_path,
    }
    path = config_dir / f"{request_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True))
    os.replace(tmp, path)                       # atomic: a rank sees whole or nothing
    return path


def read_steering_config(request_id: str, *,
                         config_dir: Path = DEFAULT_CONFIG_DIR) -> Optional[dict]:
    path = config_dir / f"{request_id}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ── attachment ────────────────────────────────────────────────────────────────

def attach_steering_layer(model: Any, layer_idx: int, *,
                          hidden_dim: Optional[int] = None
                          ) -> MetabasisSteeringLayer:
    """Replace `decoder_layers(model)[layer_idx]` with our wrapper.

    Uses the engine's OWN `decoder_layers` resolver, so the lane and the byte-exact
    engine can never disagree about which ModuleList the site indexes into.
    """
    from metabasis.extraction.hooks import decoder_layers

    layers = decoder_layers(model)
    n = len(layers)
    if not 0 <= layer_idx < n:
        raise SteeringContractViolation(
            f"layer_idx {layer_idx} out of range (model has {n} layers)")
    if hidden_dim is None:
        cfg = model.config
        hidden_dim = int(getattr(cfg, "hidden_size", None)
                         or getattr(getattr(cfg, "text_config", None), "hidden_size"))
    if isinstance(layers[layer_idx], MetabasisSteeringLayer):
        raise SteeringContractViolation(
            f"layer {layer_idx} is already wrapped -- double-wrapping would double "
            "the dose")
    wrapper = MetabasisSteeringLayer(layers[layer_idx], layer_idx, hidden_dim)
    layers[layer_idx] = wrapper
    logger.info("metabasis steering layer attached at site %d (writes layer INPUT), "
                "hidden_dim=%d", layer_idx, hidden_dim)
    return wrapper


def detach_steering_layer(model: Any, layer_idx: int) -> None:
    """Restore the original layer. A survivor across columns is a contamination."""
    from metabasis.extraction.hooks import decoder_layers

    layers = decoder_layers(model)
    w = layers[layer_idx]
    if not isinstance(w, MetabasisSteeringLayer):
        raise SteeringContractViolation(
            f"layer {layer_idx} is not wrapped; nothing to detach")
    layers[layer_idx] = w.wrapped_layer


# ── selftest (desk-side, CPU, no vLLM, no weights) ────────────────────────────

class _ToyLayer(nn.Module):
    """A decoder-layer stand-in: records its input, returns a deterministic map."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.seen: list[Tensor] = []

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        self.seen.append(hidden_states.detach().clone())
        return hidden_states * 2.0


def selftest() -> int:
    """Unit math for the four deltas and the two refusals. Returns failure count."""
    checks = 0
    fails: list[str] = []

    def ok(cond: bool, name: str) -> None:
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(name)

    torch.manual_seed(20260807)
    H, T = 16, 6
    hidden = torch.randn(1, T, H, dtype=torch.float32)
    vec = torch.randn(H, dtype=torch.float32) * 3.7        # deliberately NOT unit

    # -- 1. the delta is exactly alpha * v_hat, computed in fp32 ---------------
    alpha = 2.5
    d = steering_delta(vec, alpha, dtype=torch.float32, device=hidden.device)
    want = alpha * (vec / vec.norm())
    ok(torch.equal(d, want), "delta == alpha * v/||v|| bitwise in fp32")
    ok(abs(float(d.norm()) - alpha) < 1e-5, "||delta|| == alpha (unit vector scaled)")

    # -- 2. injection lands on the layer INPUT, exactly, at masked positions ---
    toy = _ToyLayer(H)
    w = MetabasisSteeringLayer(toy, layer_idx=3, hidden_dim=H)
    spec = MetabasisSteeringSpec(layer_idx=3, vector=vec, alpha=alpha)
    w.set_spec(spec)
    mask = torch.zeros(T, dtype=torch.bool)
    mask[3:] = True                                        # "generated" positions
    w.set_position_mask(mask)
    out = w(hidden)
    seen = toy.seen[-1]
    ok(torch.equal(seen[:, :3, :], hidden[:, :3, :]),
       "unmasked (prompt) positions reach the layer UNCHANGED")
    ok(torch.allclose(seen[:, 3:, :], hidden[:, 3:, :] + want, atol=0, rtol=0),
       "masked (generated) positions reach the layer with exactly +alpha*v_hat")
    ok(torch.equal(out, seen * 2.0), "the wrapped layer's output passes through")
    ok(w.stats["tokens_injected"] == 3 and w.stats["prefill_tokens_skipped"] == 3,
       "stats count injected vs skipped tokens")

    # -- 3. alpha == 0 is a BITWISE no-op (§2.4) -------------------------------
    toy0 = _ToyLayer(H)
    w0 = MetabasisSteeringLayer(toy0, layer_idx=0, hidden_dim=H)
    w0.set_spec(MetabasisSteeringSpec(layer_idx=0, vector=vec, alpha=0.0))
    w0.set_position_mask(torch.ones(T, dtype=torch.bool))
    w0(hidden)
    ok(torch.equal(toy0.seen[-1], hidden), "alpha=0 is bitwise identical to no hook")
    toyN = _ToyLayer(H)
    wN = MetabasisSteeringLayer(toyN, layer_idx=0, hidden_dim=H)
    wN.set_spec(None)
    wN.set_position_mask(torch.ones(T, dtype=torch.bool))
    wN(hidden)
    ok(torch.equal(toyN.seen[-1], hidden), "no spec is bitwise identical to no hook")

    # -- 4. NO RENORM: the written state's norm MOVES --------------------------
    pre = float(hidden[0, 4].norm())
    post = float(seen[0, 4].norm())
    ok(abs(pre - post) > 1e-6,
       "the injected position's norm CHANGED -- no renormalisation happened")
    ok(not hasattr(spec, "renorm") and not hasattr(spec, "intervention_type"),
       "the spec has no renorm and no intervention-type field to set")
    ok(not any(n in dir(w) for n in ("floor", "ceil", "_renormalize")),
       "the wrapper exposes no floor/ceil/renormalise operation")

    # -- 5. span rule matches hooks.py's [start_pos, None) --------------------
    pos = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.long)
    plen = torch.tensor([3, 3, 3, 3, 2, 2, 2], dtype=torch.long)
    gm = generated_position_mask(pos, plen)
    ok(gm.tolist() == [False, False, False, True, False, False, True],
       "generated_position_mask == (absolute pos >= prompt_len) per request")

    # -- 6. dose resolution + the contract's refusals --------------------------
    ok(abs(resolve_alpha(0.3, 86.2017746) - 0.3 * 86.2017746) < 1e-12,
       "resolve_alpha is alpha_frac x measured norm")
    for bad, name in (
            (lambda: resolve_alpha(0.3, 0.0), "refuses a zero §2.5 norm"),
            (lambda: MetabasisSteeringSpec(layer_idx=-1, vector=vec, alpha=1.0),
             "refuses a negative site"),
            (lambda: MetabasisSteeringSpec(layer_idx=0, vector=vec.reshape(4, 4),
                                           alpha=1.0), "refuses a non-1-D vector"),
            (lambda: MetabasisSteeringSpec(layer_idx=0, vector=vec, alpha=1.0,
                                           alpha_frac=0.3, measured_norm=2.0),
             "refuses alpha != alpha_frac x measured_norm"),
            (lambda: w.set_spec(MetabasisSteeringSpec(layer_idx=99, vector=vec,
                                                      alpha=1.0)),
             "refuses a spec whose site is not this wrapper's"),
            (lambda: measure_per_token_median_resid_norm_from_captures([]),
             "refuses to measure a norm from nothing")):
        try:
            bad()
            fails.append(name + " (did NOT raise)")
        except SteeringContractViolation:
            pass
        checks += 1

    # a wrapper asked to inject without a mask must refuse, not assume
    w2 = MetabasisSteeringLayer(_ToyLayer(H), layer_idx=1, hidden_dim=H)
    w2.set_spec(MetabasisSteeringSpec(layer_idx=1, vector=vec, alpha=1.0))
    try:
        w2(hidden)
        fails.append("missing position mask (did NOT raise)")
    except SteeringContractViolation:
        pass
    checks += 1

    # -- 7. §2.5 from captures reproduces a median of per-token norms ----------
    caps = [torch.tensor([[[3.0, 4.0], [0.0, 1.0], [6.0, 8.0]]])]   # norms 5,1,10
    ok(abs(measure_per_token_median_resid_norm_from_captures(caps) - 5.0) < 1e-9,
       "§2.5 from captures == median of per-token L2 norms")
    km = torch.tensor([True, False, True])
    ok(abs(measure_per_token_median_resid_norm_from_captures(caps, keep_mask=km)
           - 7.5) < 1e-9, "§2.5 honours a keep-mask (padding excluded)")

    # -- 8. dtype discipline: fp32 math, one cast ------------------------------
    d16 = steering_delta(vec, alpha, dtype=torch.bfloat16, device=hidden.device)
    ok(d16.dtype == torch.bfloat16, "delta is cast to the hidden dtype")
    ok(torch.equal(d16, want.to(torch.bfloat16)),
       "delta == (fp32 alpha*v_hat) cast once -- not bf16 arithmetic")

    # -- 9. attach/detach round trip on a toy stack ---------------------------
    class _ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_ToyLayer(H) for _ in range(4)])

    class _ToyOuter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _ToyModel()
            self.config = type("C", (), {"hidden_size": H})()

    m = _ToyOuter()
    orig = m.model.layers[2]
    wr = attach_steering_layer(m, 2)
    ok(isinstance(m.model.layers[2], MetabasisSteeringLayer), "attach wraps the site")
    ok(wr.wrapped_layer is orig, "the wrapper holds the ORIGINAL layer")
    try:
        attach_steering_layer(m, 2)
        fails.append("double-wrap (did NOT raise)")
    except SteeringContractViolation:
        pass
    checks += 1
    detach_steering_layer(m, 2)
    ok(m.model.layers[2] is orig, "detach restores the original layer object")

    # -- 9b. the vLLM fused-residual signature (positions, hs, residual) -------
    class _VllmToyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[tuple[Tensor, Tensor]] = []

        def forward(self, positions: Tensor, hidden_states: Tensor,
                    residual: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
            if residual is None:
                residual = hidden_states
            self.seen.append((hidden_states.detach().clone(),
                              residual.detach().clone()))
            return hidden_states, residual

    Tv = 5
    vt = _VllmToyLayer()
    wv = MetabasisSteeringLayer(vt, layer_idx=2, hidden_dim=H)
    wv.set_spec(MetabasisSteeringSpec(layer_idx=2, vector=vec, alpha=alpha))
    vmask = torch.tensor([False, False, True, True, True])
    wv.set_position_mask(vmask)
    pos = torch.arange(Tv, dtype=torch.long)
    hs2 = torch.randn(Tv, H)
    res2 = torch.randn(Tv, H)
    wv(pos, hs2, res2)
    seen_hs, seen_res = vt.seen[-1]
    ok(torch.equal(seen_hs, hs2),
       "vLLM path leaves hidden_states untouched (the accumulator is written)")
    ok(torch.equal(seen_res[:2], res2[:2]),
       "vLLM path: prompt positions of the residual are UNCHANGED")
    ok(torch.allclose(seen_res[2:], res2[2:] + want, atol=0, rtol=0),
       "vLLM path: generated positions of the residual get exactly +alpha*v_hat")
    ok(abs(float((seen_hs + seen_res)[3].norm())
           - float((hs2 + res2 + want)[3].norm())) < 1e-4,
       "vLLM path: the SUM (the real residual stream) carries the delta once")
    vt0 = _VllmToyLayer()
    wv0 = MetabasisSteeringLayer(vt0, layer_idx=0, hidden_dim=H)
    wv0.set_spec(MetabasisSteeringSpec(layer_idx=0, vector=vec, alpha=alpha))
    wv0.set_position_mask(vmask)
    wv0(pos, hs2, None)
    ok(torch.allclose(vt0.seen[-1][0][2:], hs2[2:] + want, atol=0, rtol=0),
       "vLLM first layer (residual=None): hidden_states IS the residual stream")

    # -- 10. capture is exactly the tensor that gets written (the proof hook) --
    toyc = _ToyLayer(H)
    wc = MetabasisSteeringLayer(toyc, layer_idx=0, hidden_dim=H)
    wc.set_spec(None)
    wc.set_position_mask(torch.zeros(T, dtype=torch.bool))
    wc.capture_input_hidden_states(True)
    wc(hidden)
    caps2 = wc.take_captures()
    ok(len(caps2) == 1 and torch.equal(caps2[0], hidden),
       "capture_input_hidden_states records the layer-INPUT tensor by value")

    print(f"vllm_residual_write selftest checks run: {checks} "
          f"({len(fails)} failure(s))")
    for f in fails:
        print("  MISS " + f)
    return len(fails)


if __name__ == "__main__":
    import sys
    sys.exit(1 if selftest() else 0)
