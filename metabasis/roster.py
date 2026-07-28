"""Roster nodes and their site-SCAN grids — the registry of what was scanned.

Two registries, deliberately separate (do not merge them), and they OVERLAP:

- `fit_transport_maps.SITES` — the **fixed fit grid**. A model appears there
  only after the desk has ratified its site of record from an alignment-curve
  scan (or it is a carried, banked model). Membership means "we know where
  this model's sites are", and the fit CLIs will run it with no `--src-sites`
  / `--tgt-sites` override.
- `SCAN_GRIDS` here — the **scan grid** each roster node was collected and
  curve-scanned on: the computed 12 sites, plus any ruled extension. Membership
  means "this grid is what we looked at".

A node may also carry deviations that travel with it wherever it is used:
`max_seq_len` (a position-ceiling truncation, prereg ADDENDUM 2026-07-27-B),
`scan_grid_extension` (ruled extra sites when the default window failed to
bracket the peak), and `blocked_reason` (collection forbidden until a ruling).
They live here, not in job scripts, because every use of a node — collection,
spot-replay, target build, behavioral read — has to agree about them.

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

from pydantic import BaseModel, Field, field_validator, model_validator

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
    max_position_embeddings: int | None = Field(
        default=None,
        description="the checkpoint's positional capacity. For LEARNED absolute "
                    "position embeddings (GPT-2 lineage) this is a HARD ceiling: a "
                    "longer sequence indexes past the `wpe` table. Recorded so the "
                    "job preflight's corpus-length check is mechanical, not folklore.")
    scan_grid_extension: tuple[int, ...] = Field(
        default=(),
        description="Ruled additional sites, appended to the computed 12-site grid. "
                    "A DEVIATION FROM THE DEFAULT INSTRUMENT, only ever set by an "
                    "explicit desk/Luxia ruling recorded in the ledger, and only for "
                    "the reason the ruling names — the [0.15,0.85] 12-site rule stands "
                    "as the default. Kept separate from the computed grid so the rule "
                    "and the exception never blur: `scan_grid` is always re-derivable "
                    "from num_hidden_layers alone.")
    max_seq_len: int | None = Field(
        default=None,
        description="prereg ADDENDUM 2026-07-27-B: if set, every use of this node "
                    "truncates each text to its FIRST `max_seq_len` tokens "
                    "(`--max-seq-len`). Lives in the registry rather than in a job "
                    "script so collection, spot-replay, target builds and behavioral "
                    "reads cannot silently disagree — the addendum requires ONE "
                    "truncation for all uses.")
    blocked_reason: str | None = Field(
        default=None,
        description="non-None => this node MUST NOT be collected. Set when a "
                    "verified, non-negotiable blocker exists (e.g. the frozen corpus "
                    "does not fit the architecture). Job preflights assert this is "
                    "None; the desk clears it only with an explicit ruling.")
    notes: str = ""

    @property
    def is_blocked(self) -> bool:
        return self.blocked_reason is not None

    @property
    def collector_args(self) -> list[str]:
        """The deviation-carrying flags every invocation of this node MUST pass.
        Job scripts splice this in rather than re-typing the numbers, so collection
        and spot-replay cannot drift apart (they would compare different objects and
        the bitwise gate would fail for the wrong reason)."""
        return ["--max-seq-len", str(self.max_seq_len)] if self.max_seq_len else []

    @model_validator(mode="after")
    def _ceiling_is_respected(self) -> "RosterNode":
        if (self.max_seq_len and self.max_position_embeddings
                and self.max_seq_len > self.max_position_embeddings):
            raise ValueError(
                f"{self.key}: max_seq_len={self.max_seq_len} exceeds the "
                f"architecture's positional capacity {self.max_position_embeddings}")
        return self

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
    def effective_scan_grid(self) -> tuple[int, ...]:
        """The grid actually collected: the computed rule plus any ruled extension."""
        return tuple(sorted(set(self.scan_grid) | set(self.scan_grid_extension)))

    @property
    def sites_arg(self) -> str:
        """The `--sites` value for `collect_mean_states.py`."""
        return ",".join(str(s) for s in self.effective_scan_grid)

    @model_validator(mode="after")
    def _extension_in_range(self) -> "RosterNode":
        bad = [s for s in self.scan_grid_extension
               if not 0 <= s < self.num_hidden_layers]
        if bad:
            raise ValueError(
                f"{self.key}: extension sites {bad} outside [0, "
                f"{self.num_hidden_layers - 1}] — site L is a hook on "
                f"decoder_layers[L], so num_hidden_layers-1 is the valid maximum")
        return self


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

# Hub rungs 2 (2026-07-27): three more pairs to firm the star constants before the
# first prediction batch. Architecture facts read from each checkpoint's own
# config.json on 2026-07-27 (shas below), on the collection node, in the collection
# venv (transformers 5.3.0); checkpoint identity verified per rake M9 — eos ids +
# generation_config + chat-template sha, NEVER the directory name.
HUB_RUNGS_2: tuple[RosterNode, ...] = (
    RosterNode(
        key="pythia-6.9b", model_id="EleutherAI/pythia-6.9b",
        roster_row=16, arms=("raw",), num_hidden_layers=32, hidden_size=4096,
        weights_dirname="pythia-6.9b", checkpoint_identity="base",
        config_sha256="d7f2d0bbfa279e3324423a2882d8fd1e1276fce7806a152cf5f5648733fdccab",
        max_position_embeddings=2048,
        scan_grid_extension=(28, 29, 30, 31),
        notes="RULED GRID EXTENSION (Luxia 2026-07-27, ledger row: window-failed-to-"
              "bracket). The computed 12-site [0.15,0.85] grid produced NO INTERIOR "
              "PEAK on the record arm — held-out r2 rose monotonically from L5 to the "
              "top site L27 from all three hub sources (8bL18->L27 r2=.1523, still "
              "climbing), so the window failed to bracket the peak and 'sites from "
              "curves, never fiat' could not be satisfied inside it. Extended by the "
              "model's remaining depth L28-L31 (L31 = num_hidden_layers-1, the valid "
              "maximum). The 12-site rule remains the DEFAULT instrument; this "
              "extension is the named exception, not a new rule. "
              "GPTNeoXForCausalLM — the FIRST non-`.model.layers` architecture on the "
              "roster; resolves as `.gpt_neox.layers` via base_model_prefix (hooks.py "
              "fallback added 2026-07-27, CPU-verified against output_hidden_states). "
              "BASE: no chat template, no generation_config, bos == eos == 0 — raw arm "
              "only, per prereg row 16. Rotary (rotary_pct=0.25), so 2048 is a soft "
              "context, but the corpus tops out at 1526 raw tokens anyway (headroom "
              "522, all 780 texts fit). NAMED DTYPE NOTE: config torch_dtype is "
              "float16 (Pythia was trained fp16) while the collector forwards in "
              "bfloat16 like every other node — uniform campaign treatment, and the "
              "bitwise spot-replay gate certifies determinism, but bf16 has 3 fewer "
              "mantissa bits than fp16, so this is a real (if uniform) cast."),
    RosterNode(
        key="gpt2-xl", model_id="openai-community/gpt2-xl",
        roster_row=17, arms=("raw",), num_hidden_layers=48, hidden_size=1600,
        weights_dirname="gpt2-xl", checkpoint_identity="base",
        config_sha256="be48115d52314bc27327e160bf10197760db02ea689d10adb24db4102edf7a8a",
        max_position_embeddings=1024,
        max_seq_len=1024,
        scan_grid_extension=(0, 1, 2, 3, 4, 5, 6, 42, 43, 44, 45, 46, 47),
        notes="RULED BOTH-EDGE EXTENSION (Luxia 2026-07-27): the computed 12-site "
              "curve peaked at L7, its LOWEST site (r²=.449/.423/.450 from the three "
              "hub sources), collapsed to a trough at L13–16 (≈.055), then climbed "
              "monotonically to L41 (≈.375) — i.e. it is edge-peaked at BOTH ends and "
              "brackets nothing. Extended by L0–L6 and L42–L47 for the complete "
              "25-site curve; L0 (the residual ENTERING layer 0, i.e. the embedding "
              "output) is deliberately included — the question is where the "
              "surface-alignment spike goes, and that cannot be answered by stopping "
              "short of it. gpt2-xl STAYS SCAN-REGISTRY: no site is picked until the "
              "desk reads the completed curve. "
              "POSITION-CEILING DEVIATION, prereg ADDENDUM 2026-07-27-B (ratified by "
              "Luxia 2026-07-27): GPT-2 uses LEARNED absolute position embeddings "
              "(`wpe`, n_positions=1024) — a hard ceiling on every GPT-2 variant, not "
              "a soft one — and 148 of the 780 frozen corpus texts exceed it under "
              "GPT-2's own BPE (max 1565, mean 643.9). This node therefore collects "
              "with per-text truncation to its FIRST 1024 tokens (`--max-seq-len "
              "1024`), applied IDENTICALLY to every use of the node, recorded in every "
              "stamp as `truncation: first-1024`. The other 632 texts are "
              "byte-identical to the frozen objects (verified: suffix-drop only, and "
              "collection is batch-1 so no text perturbs another). NAMED CAVEAT, "
              "frozen with the deviation: for those 148 texts any gpt2-xl pair "
              "compares full-text mean states against truncated-text mean states — a "
              "measurement-basis mismatch confined to this node, flagged at scoring. "
              "GPT2LMHeadModel — resolves as `.transformer.h` (the GPT-2 lineage names "
              "its decoder stack `h`, not `layers`); hooks.py fallback covers it and "
              "the capture was CPU-verified bit-identical to output_hidden_states, so "
              "the hook path was CPU-verified bit-identical to output_hidden_states. "
              "Row 17 is the deliberate floor-stress rung (d=1600, oldest diet). BASE: "
              "no chat template, bos == eos == 50256, raw arm only. Weights are fp32 "
              "on disk (config carries no torch_dtype); the collector forwards bf16."),
    RosterNode(
        key="llama-3.1-70b-instruct", model_id="meta-llama/Llama-3.1-70B-Instruct",
        roster_row=11, arms=("native", "raw"), num_hidden_layers=80, hidden_size=8192,
        weights_dirname="llama-3.1-70b-instruct", checkpoint_identity="instruct",
        config_sha256="fa6e9124e4621df77aecf96fbfaf7975814013d2d5ab1c972e965000588a9749",
        max_position_embeddings=131072,
        notes="The scale ceiling of the lineage-matched ladder, and the first node "
              "collected through the SHARDED path (`--shard-across 2 "
              "--assert-multi-device`; cert PASSED 2026-07-27, ledger "
              "collection/shard-cert). ~140 GiB bf16 does not fit one card beside the "
              "~66 GiB foreign resident. NEVER --allow-offload (standing desk ruling). "
              "IDENTITY (rake M9, verified independently of the dirname): eos ids "
              "[128001, 128008, 128009] and generation_config temperature 0.6 / "
              "top_p 0.9 — the instruct signature; chat-template sha e10ca381… is "
              "IDENTICAL to the banked 8B hub, so the native arm is lineage-matched to "
              "the hub by construction. NOT to be confused with the sibling BASE "
              "`Llama-3.1-70B` directory on the shared store, which is QUARANTINED "
              "(rake M9: another user's hand-written chat template)."),
)

ROSTER: dict[str, RosterNode] = {n.key: n for n in WAVE1 + HUB_RUNGS_2}

#: model key -> the grid it was actually collected and curve-scanned on. This is
#: what `--sites` should carry for a scan collection, and what `--tgt-sites`
#: should carry for the scan fit. Normally the computed 12 sites; for a node with
#: a ruled extension (pythia-6.9b) it is the EFFECTIVE grid, because that is what
#: the bank holds and what reproduces the published curve.
SCAN_GRIDS: dict[str, tuple[int, ...]] = {
    k: n.effective_scan_grid for k, n in ROSTER.items()}


def scan_grid_table() -> str:
    """The desk-ratification table: model · row · arms · n_layers · grid."""
    w = max(len(k) for k in ROSTER)
    head = (f"{'model key':<{w}}  row  arms         n_layers  scan grid "
            f"({SCAN_N_SITES} sites, depth {SCAN_DEPTH_LO}–{SCAN_DEPTH_HI})")
    rows = [head, "-" * len(head)]
    for key, node in sorted(ROSTER.items()):
        g = node.scan_grid
        # The computed grid is printed as the rule produced it; a ruled extension
        # is printed SEPARATELY and labelled, so the instrument and the exception
        # are never mistaken for one another at a glance.
        rows.append(f"{key:<{w}}  {node.roster_row:>3}  "
                    f"{'+'.join(node.arms):<12} {node.num_hidden_layers:>8}  "
                    f"{','.join(str(s) for s in g)}"
                    + ("" if len(g) == SCAN_N_SITES else f"   [DEDUPED to {len(g)}]")
                    + (f"   + RULED {','.join(str(s) for s in node.scan_grid_extension)}"
                       if node.scan_grid_extension else "")
                    + ("   ** BLOCKED **" if node.is_blocked else ""))
    blocked = blocked_nodes()
    if blocked:
        rows.append("")
        for key, reason in sorted(blocked.items()):
            rows.append(f"BLOCKED {key}: {reason}")
    return "\n".join(rows)


def blocked_nodes() -> dict[str, str]:
    """key -> blocker, for every node that must not be collected. Job preflights
    call this and refuse to run; empty dict means the whole roster is clear."""
    return {k: n.blocked_reason for k, n in ROSTER.items() if n.blocked_reason}


def collectable(keys: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """The subset of `keys` (default: the whole roster) that is safe to collect."""
    return tuple(sorted(k for k in (keys or tuple(ROSTER))
                        if not ROSTER[k].is_blocked))


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
        # EFFECTIVE grid, not the computed one: a ruled extension (registry-
        # recorded, ledger-backed) is part of the curve we actually scanned, so a
        # peak found there is still "from the curve". Checking against the bare
        # computed grid would flag a legitimately-ratified extension site as fiat.
        scanned = node.effective_scan_grid
        if fixed is None or set(fixed).issubset(scanned):
            continue
        stray = sorted(set(fixed) - set(scanned))
        problems.append(
            f"{key}: fixed fit grid {tuple(fixed)} contains site(s) {stray} that "
            f"its own scan grid {scanned} never visited — either a FIAT "
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
        print(f"  {key:<26} fit grid {grid}  <- scan grid "
              f"{ROSTER[key].effective_scan_grid}")
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
    print(f"collectable now:          {list(collectable())}")
