"""Roster nodes and their site-SCAN grids — the registry for models whose fit
grid is not fixed yet.

Two registries, deliberately separate (do not merge them):

- `fit_transport_maps.SITES` — the **fixed fit grid**. A model appears there
  only after the desk has ratified its site of record from an alignment-curve
  scan (or it is a carried, banked model). Membership means "we know where
  this model's sites are".
- `SCAN_GRIDS` here — the **12-site scan grid**, the thing a new roster node
  gets *before* its fit grid exists. Membership means "we are still looking".

The prereg (frozen, `PREREG-transport-campaign-2026-07-26.md` §4) says:
*"Sites from curves, never fiat: every NEW model gets an alignment-curve site
scan (12-site grid, pick the peak — the OLMo precedent) before its fit grid is
fixed."* This module is that rule as code — the grid is COMPUTED from
`num_hidden_layers` by `scan_grid()`, never typed in by hand, so any reader can
re-derive it from a checkpoint's `config.json`.

The OLMo precedent, for the shape of the read this grid feeds:
`smalls/fits_olmo/cp2_summary.json` — hub 8B at its banked sites (L14/16/18)
× the candidate's grid, all four fit families, both arms; the alignment curve
is held-out r² at `proc_k128` per site-pair, with `r2_null_shuffled_q95` beside
and `cka_after` secondary. OLMo-2-7B (base) peaked at L16, r²=.6445 from 8bL14.

Weights paths are NEVER hardcoded here: `weights_dirname` is the *basename*
convention (`/models/<Name>` on the collection node), and the actual directory
is passed to the collector via `--model-path`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- the scan rule
SCAN_DEPTH_LO = 0.15
SCAN_DEPTH_HI = 0.85
SCAN_N_SITES = 12


def scan_grid(num_hidden_layers: int, lo: float = SCAN_DEPTH_LO,
              hi: float = SCAN_DEPTH_HI, n: int = SCAN_N_SITES) -> tuple[int, ...]:
    """The frozen scan-grid rule: `n` sites at even fractional depth over
    [lo, hi] of `num_hidden_layers`, rounded to layer indices, deduped, sorted.

    Sites are decoder-layer indices in the collector's convention — site L is a
    `forward_pre_hook` on `decoder_layers(model)[L]`, i.e. the residual
    ENTERING layer L — so the valid range is [0, num_hidden_layers - 1].

    Deduping can return FEWER than `n` sites on very shallow models; callers
    that care should check `len()`. All wave-1 nodes (32–64 layers) yield 12.
    """
    if num_hidden_layers < 3:
        raise ValueError(f"num_hidden_layers={num_hidden_layers}: too shallow to scan")
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"bad depth window [{lo}, {hi}]")
    if n < 2:
        raise ValueError(f"n={n}: need at least 2 sites")
    fracs = (lo + i * (hi - lo) / (n - 1) for i in range(n))
    sites = {min(max(int(round(f * num_hidden_layers)), 0), num_hidden_layers - 1)
             for f in fracs}
    return tuple(sorted(sites))


# ---------------------------------------------------------------- node records
class RosterNode(BaseModel):
    """One roster model that still needs a site scan.

    Every architecture field is verified against the checkpoint's own
    `config.json` (sha recorded) — the same discipline `config.MODEL_PRESETS`
    documents. `checkpoint_identity` is verified against chat-template
    presence + eos ids + README, NOT assumed from the repo name (see the
    Qwen3-30B-A3B-Base and Llama-3.1-70B findings in the wave-1 audit).
    """

    key: str = Field(description="bank key; keys the state banks and every path")
    model_id: str = Field(description="HF repo id (the checkpoint of record)")
    roster_row: int = Field(description="row in the frozen prereg roster table")
    arms: tuple[str, ...] = Field(description="template arms this node runs")
    num_hidden_layers: int
    hidden_size: int
    weights_dirname: str = Field(
        description="basename convention under the collection node's weights root "
                    "(the full path is passed via --model-path; never hardcoded)")
    checkpoint_identity: Literal["instruct", "base"]
    config_sha256: str | None = Field(
        default=None, description="sha256 of the config.json these numbers came from")
    notes: str = ""

    @field_validator("arms")
    @classmethod
    def _known_arms(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        bad = [a for a in v if a not in ("native", "raw")]
        if bad:
            raise ValueError(f"unknown arms {bad}")
        if not v:
            raise ValueError("a node must run at least one arm")
        return v

    @property
    def scan_grid(self) -> tuple[int, ...]:
        return scan_grid(self.num_hidden_layers)

    @property
    def sites_arg(self) -> str:
        """The `--sites` value for `collect_mean_states.py`."""
        return ",".join(str(s) for s in self.scan_grid)


# Wave 1 of the collection phase: the seven cheap new nodes. Architecture facts
# read from each checkpoint's own config.json on 2026-07-26 (shas below);
# chat-template / eos / arm facts verified at build_ids level on the same date.
WAVE1: tuple[RosterNode, ...] = (
    RosterNode(
        key="olmo2-7b-instruct", model_id="allenai/OLMo-2-1124-7B-Instruct",
        roster_row=7, arms=("native", "raw"), num_hidden_layers=32, hidden_size=4096,
        weights_dirname="OLMo-2-1124-7B-Instruct", checkpoint_identity="instruct",
        config_sha256="ff8cc8709a229515676797ab6f343a09391041c9a8fbbc78bfec5be4c2e3664e",
        notes="The instruct-OLMo discriminator against the banked BASE `olmo2-7b` "
              "(same pretrain, chat template added). Distinct bank key — check any "
              "glob over states_olmo2-7b*, which now matches BOTH banks. "
              "bos == eos == 100257; the native template opens with <|endoftext|>."),
    RosterNode(
        key="qwen2.5-3b-instruct", model_id="Qwen/Qwen2.5-3B-Instruct",
        roster_row=8, arms=("native", "raw"), num_hidden_layers=36, hidden_size=2048,
        weights_dirname="Qwen2.5-3B-Instruct", checkpoint_identity="instruct",
        config_sha256="eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b",
        notes="tie_word_embeddings=True. Template injects Qwen's DEFAULT system "
              "prompt when the entry carries none (deterministic, pinned by the "
              "chat-template sha)."),
    RosterNode(
        key="qwen2.5-14b-instruct", model_id="Qwen/Qwen2.5-14B-Instruct",
        roster_row=9, arms=("native", "raw"), num_hidden_layers=48, hidden_size=5120,
        weights_dirname="Qwen2.5-14B-Instruct", checkpoint_identity="instruct",
        config_sha256="0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e",
        notes="Same chat template as the 3B/32B rungs (sha cd8e9439…)."),
    RosterNode(
        key="qwen2.5-32b-instruct", model_id="Qwen/Qwen2.5-32B-Instruct",
        roster_row=10, arms=("native", "raw"), num_hidden_layers=64, hidden_size=5120,
        weights_dirname="Qwen2.5-32B-Instruct", checkpoint_identity="instruct",
        config_sha256="9c6772f138ef9e5b3d1c18f2c87e451bbc01f5f1a4eabb36f9bf4f53829b903e",
        notes="Only wave-1 node whose weights were already on the collection node."),
    RosterNode(
        key="mistral-7b-instruct-v0.3", model_id="mistralai/Mistral-7B-Instruct-v0.3",
        roster_row=13, arms=("native", "raw"), num_hidden_layers=32, hidden_size=4096,
        weights_dirname="Mistral-7B-Instruct-v0.3", checkpoint_identity="instruct",
        config_sha256="affafc6478ec0fd07a32f0ca57aa2fc57743f4d17d6730f86a96ac24d1507f99",
        notes="Gated HF repo. Raw-arm specials prefix is [1] (<s>); the v0.3 "
              "template folds a system message into the first [INST] block, so all "
              "780 corpus entries template cleanly."),
    RosterNode(
        key="phi-4", model_id="microsoft/phi-4",
        roster_row=14, arms=("native", "raw"), num_hidden_layers=40, hidden_size=5120,
        weights_dirname="phi-4", checkpoint_identity="instruct",
        config_sha256="07eedad2c48798b6e3728e4a1b75e0e092019a375ba725a40e40c78d13664045",
        notes="Phi3ForCausalLM architecture, im_start/im_sep/im_end template. "
              "Raw arm adds NO specials prefix (prefix length 0)."),
    RosterNode(
        key="phi-3.5-mini-instruct", model_id="microsoft/Phi-3.5-mini-instruct",
        roster_row=15, arms=("native", "raw"), num_hidden_layers=32, hidden_size=3072,
        weights_dirname="Phi-3.5-mini-instruct", checkpoint_identity="instruct",
        config_sha256="224de4f6a15b9d2a89695ec04b7f7ab2dd93a008a506979925e5a88cb5804974",
        notes="head_dim 96 (the only wave-1 node that is not 128). Raw arm adds NO "
              "specials prefix despite bos_token_id=1 (add_bos_token is off), and "
              "the native template does not open with <s> either — the two arms "
              "agree, which is what the arm-consistency rule cares about."),
)

ROSTER: dict[str, RosterNode] = {n.key: n for n in WAVE1}

#: model key -> its 12-site scan grid. This is what `--sites` should carry for a
#: scan collection, and what `--tgt-sites` should carry for the scan fit.
SCAN_GRIDS: dict[str, tuple[int, ...]] = {k: n.scan_grid for k, n in ROSTER.items()}


def scan_grid_table() -> str:
    """The desk-ratification table: model · row · n_layers · grid."""
    w = max(len(k) for k in ROSTER)
    head = (f"{'model key':<{w}}  row  n_layers  scan grid "
            f"({SCAN_N_SITES} sites, depth {SCAN_DEPTH_LO}–{SCAN_DEPTH_HI})")
    rows = [head, "-" * len(head)]
    for key, node in sorted(ROSTER.items()):
        g = node.scan_grid
        rows.append(f"{key:<{w}}  {node.roster_row:>3}  {node.num_hidden_layers:>8}  "
                    f"{','.join(str(s) for s in g)}"
                    + ("" if len(g) == SCAN_N_SITES else f"   [DEDUPED to {len(g)}]"))
    return "\n".join(rows)


def check_key_collisions(existing_keys: set[str]) -> list[str]:
    """Bank-naming collision guard — state banks are keyed by model key.

    Returns the offending keys (exact collisions, and prefix relations, which
    are what break careless globs like `states_olmo2-7b*`).
    """
    problems: list[str] = []
    for key in ROSTER:
        if key in existing_keys:
            problems.append(f"{key}: EXACT collision with an existing bank key")
        for other in existing_keys:
            if key != other and (key.startswith(other) or other.startswith(key)):
                problems.append(
                    f"{key}: prefix relation with existing bank key {other!r} — "
                    f"exact-name loads are safe, GLOBS are not")
    return problems


if __name__ == "__main__":                                   # desk convenience
    from metabasis.scripts.fit_transport_maps import SITES

    print(scan_grid_table())
    print()
    for line in check_key_collisions(set(SITES)):
        print(f"COLLISION-GUARD: {line}")
    print(f"\nfixed fit grids (SITES): {sorted(SITES)}")
    print(f"scan grids (SCAN_GRIDS): {sorted(SCAN_GRIDS)}")
