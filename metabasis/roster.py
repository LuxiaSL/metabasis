"""Roster nodes and their site-SCAN grids — the registry of what was scanned.

Two registries, deliberately separate (do not merge them), and they OVERLAP:

- `fit_transport_maps.SITES` — the **fixed fit grid**. A model appears there
  only after the desk has ratified its site of record from an alignment-curve
  scan (or it is a carried, banked model). Membership means "we know where
  this model's sites are", and the fit CLIs will run it with no `--src-sites`
  / `--tgt-sites` override.
- `SCAN_GRIDS` here — the **12-site scan grid** each roster node was collected
  and curve-scanned on. Membership means "this grid is what we looked at".

Before ratification a node is in SCAN_GRIDS ONLY: "we are still looking", and
every run must pass its grid explicitly. Ratification does not MOVE the node —
it ADDS the key to `SITES` and leaves it here, so the scan that produced the
site of record stays re-derivable from the same registry that drove it. The
wave-1 seven (ratified by Luxia 2026-07-27) are in both. `audit_registries()`
is the guard for that overlap: a key in both registries must have a fixed grid
drawn FROM its own scan grid (sites from curves, never fiat), and prefix
relations between bank keys stay flagged either way.

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

from collections.abc import Mapping
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


def fiat_grid_problems(fixed_grids: Mapping[str, tuple[int, ...]]) -> list[str]:
    """The ratification invariant: a graduated node's fixed fit grid is drawn
    from its OWN scan grid (prereg §4, "sites from curves, never fiat").

    A violation means one of two things, both fatal: the grid was typed in by
    fiat, or a DIFFERENT model is wearing this bank key. Cheap enough to run at
    import time, which is where `fit_transport_maps` runs it.
    """
    problems: list[str] = []
    for key, node in ROSTER.items():
        fixed = fixed_grids.get(key)
        if fixed is None or set(fixed).issubset(node.scan_grid):
            continue
        stray = sorted(set(fixed) - set(node.scan_grid))
        problems.append(
            f"{key}: fixed fit grid {tuple(fixed)} contains site(s) {stray} that "
            f"its own scan grid {node.scan_grid} never visited — either a FIAT "
            f"grid (prereg §4 forbids it) or a different model reusing this key")
    return problems


def unexplained_collisions(existing_keys: set[str],
                           ratified_grids: Mapping[str, tuple[int, ...]]) -> list[str]:
    """EXACT bank-key collisions that ratification does not explain.

    The key IS the bank identity, so a roster key equal to a key already in use
    is only benign when the two are the SAME model — i.e. the node graduated
    into the fixed-fit-grid registry. Pass that registry as `ratified_grids`
    and graduations are excused here; `fiat_grid_problems()` is what checks
    that an excused key really did take its grid from its own scan curve.
    """
    return [f"{key}: EXACT collision with an existing bank key"
            for key in ROSTER if key in existing_keys and key not in ratified_grids]


def prefix_hazards(existing_keys: set[str]) -> list[str]:
    """Bank keys in a prefix relation — glob hazards, ratified or not.

    Expected and permanent for the OLMo pair (`olmo2-7b` BASE beside
    `olmo2-7b-instruct`): exact-name loads are safe, `states_olmo2-7b*` is not.
    Reported so nobody writes that glob, never treated as a fault to fix.
    """
    return [f"{key}: prefix relation with existing bank key {other!r} — "
            f"exact-name loads are safe, GLOBS are not"
            for key in ROSTER for other in existing_keys
            if key != other and (key.startswith(other) or other.startswith(key))]


def check_key_collisions(existing_keys: set[str],
                         ratified_grids: Mapping[str, tuple[int, ...]] | None = None
                         ) -> list[str]:
    """Bank-naming collision guard — every reportable line, in one list.

    Fiat grids first, then unexplained exact collisions, then prefix hazards.
    Callers that need to tell fatal from expected should use
    `audit_registries()` instead of splitting these strings.
    """
    ratified = ratified_grids or {}
    return (fiat_grid_problems(ratified)
            + unexplained_collisions(existing_keys, ratified)
            + prefix_hazards(existing_keys))


class RegistryAudit(BaseModel):
    """The state of the two registries at one moment, as data rather than prose."""

    graduated: dict[str, tuple[int, ...]] = Field(
        default_factory=dict,
        description="roster key -> ratified fixed fit grid (in BOTH registries)")
    still_scanning: tuple[str, ...] = Field(
        default=(), description="roster keys with no fixed fit grid yet")
    fatal: tuple[str, ...] = Field(
        default=(), description="fiat grids + exact collisions ratification cannot "
                                "explain — a build/naming bug, fix before fitting")
    expected: tuple[str, ...] = Field(
        default=(), description="prefix hazards — permanent facts about the bank "
                                "namespace, reported so globs stay unwritten")

    @property
    def problems(self) -> tuple[str, ...]:
        """Every reportable line, fatal first (display order)."""
        return self.fatal + self.expected

    @property
    def ok(self) -> bool:
        return not self.fatal


def audit_registries(fixed_grids: Mapping[str, tuple[int, ...]]) -> RegistryAudit:
    """Cross-check the scan registry here against a fixed-fit-grid registry.

    `fixed_grids` is `fit_transport_maps.SITES` (passed in rather than imported,
    so this module stays import-cycle-free and unit-testable against a stub).
    """
    keys = set(fixed_grids)
    return RegistryAudit(
        graduated={k: tuple(fixed_grids[k]) for k in sorted(ROSTER) if k in fixed_grids},
        still_scanning=tuple(k for k in sorted(ROSTER) if k not in fixed_grids),
        fatal=tuple(fiat_grid_problems(fixed_grids)
                    + unexplained_collisions(keys, fixed_grids)),
        expected=tuple(prefix_hazards(keys)))


if __name__ == "__main__":                                   # desk convenience
    from metabasis.scripts.fit_transport_maps import SITES

    print(scan_grid_table())
    audit = audit_registries(SITES)
    print(f"\ngraduated (fixed fit grid ratified; in BOTH registries): "
          f"{len(audit.graduated)}/{len(ROSTER)}")
    for key, grid in audit.graduated.items():
        print(f"  {key:<26} fit grid {grid}  <- scan grid {ROSTER[key].scan_grid}")
    print(f"still scanning (no fixed grid; pass --src-sites/--tgt-sites): "
          f"{list(audit.still_scanning)}")
    print()
    for line in audit.fatal:
        print(f"COLLISION-GUARD [FATAL]: {line}")
    for line in audit.expected:
        print(f"COLLISION-GUARD [expected]: {line}")
    print(f"registry audit: {'OK' if audit.ok else 'FATAL — fix before fitting'}")
    print(f"\nfixed fit grids (SITES): {sorted(SITES)}")
    print(f"scan grids (SCAN_GRIDS): {sorted(SCAN_GRIDS)}")
