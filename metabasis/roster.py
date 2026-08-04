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
convention (a fixed weights root on the collection node), and the actual directory
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

# The MoE chat rung (2026-07-28): roster row 18. Architecture facts read from the
# checkpoint's own config.json on the collection node; checkpoint identity verified
# per rake M9 — eos ids + generation_config + HUB CROSS-CHECK of the config sha,
# NEVER the directory name (the sibling `Qwen3-30B-A3B-Base` directory on the same
# shared store is a DIFFERENT checkpoint and does not satisfy row 18).
MOE_CHAT: tuple[RosterNode, ...] = (
    RosterNode(
        key="qwen3-30b-a3b", model_id="Qwen/Qwen3-30B-A3B",
        roster_row=18, arms=("native", "raw"), num_hidden_layers=48, hidden_size=2048,
        weights_dirname="Qwen3-30B-A3B", checkpoint_identity="instruct",
        config_sha256="2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297",
        max_position_embeddings=40960,
        notes="The MoE rung: Qwen3MoeForCausalLM, 128 experts, top-8 routing "
              "(num_experts_per_tok=8), moe_intermediate_size 768, decoder_sparse_step=1 "
              "(EVERY layer is an MoE layer, mlp_only_layers=[]). 30.5B total params / "
              "~3.3B active — 56.87 GiB bf16 on disk, which is why it collects "
              "single-card despite the 30B nameplate. hidden_size 2048 is the SMALLEST "
              "residual width on the roster above gpt2-xl, and it is the width the "
              "transport map sees: a 30B-nameplate node with a 2048-d residual stream "
              "is the sharpest available test of whether the exchange rate tracks "
              "capability or width. "
              "IDENTITY (rake M9, verified independently of the dirname): config.json "
              "sha 2850ddb3… is byte-identical to the hub's Qwen/Qwen3-30B-A3B, while "
              "the sibling BASE repo's config hashes 7e414215… — different object. "
              "eos_token_id 151645 (<|im_end|>) is the CHAT signature; the Base "
              "checkpoint carries 151643 (<|endoftext|>). generation_config ships the "
              "chat sampling defaults (temperature 0.6 / top_k 20 / top_p 0.95, eos "
              "[151645, 151643]). Chat template present in tokenizer_config — recorded, "
              "but per rake M9 template presence proves NOTHING on its own (Qwen ships "
              "templates on Base checkpoints too); the eos id + config sha are the "
              "discriminators of record. This closes the wave-1 audit finding that "
              "the sibling Base directory on the shared weight store does not satisfy roster row 18. "
              "Note the roster row asks for native+raw, i.e. a chat model — satisfied "
              "by this checkpoint and by no other Qwen3-30B-A3B directory on the node."),
)

# The big-chain single-card rungs (2026-07-28): roster rows 12 and 19, the two
# members of the big-chain pull that fit ONE card. Architecture facts read from
# each checkpoint's own config.json on the shared weight store; identity verified
# per rake M9 — config.json sha CROSS-CHECKED AGAINST THE HUB, plus eos ids and
# generation_config, NEVER the directory name and never template presence.
# The other two big-chain repos (rows 20 and 21, Llama-3.1-405B-Instruct and
# DeepSeek-V3) are downloaded and M9-verified but DELIBERATELY ABSENT here: both
# are multicard-only, and the multicard scan design is held for a ruling. Adding
# a key here is what makes a node collectable, so their absence is the block.
BIG_CHAIN_SINGLE_CARD: tuple[RosterNode, ...] = (
    RosterNode(
        key="llama-3.3-70b-instruct", model_id="meta-llama/Llama-3.3-70B-Instruct",
        roster_row=12, arms=("native", "raw"), num_hidden_layers=80, hidden_size=8192,
        weights_dirname="Llama-3.3-70B-Instruct", checkpoint_identity="instruct",
        config_sha256="95ef9768e4741543dbfaf0c274f101855883ff338b235c99eca2b6a4f4abee12",
        max_position_embeddings=131072,
        notes="THE POST-TRAINING-VINTAGE ROW. Row 12 exists to be compared against "
              "row 11 (llama-3.1-70b-instruct), and the comparison is unusually clean: "
              "the two checkpoints are architecturally IDENTICAL — LlamaForCausalLM, "
              "80 layers, hidden_size 8192, 64 heads, head_dim 128, intermediate 28672, "
              "vocab 128256, the same llama3 rope_scaling (factor 8.0, original "
              "max_position_embeddings 8192) — so the ONLY variable between them is the "
              "RLHF era. Because both are 80 layers the computed 12-site scan grids are "
              "the SAME grid, and the two alignment curves are therefore comparable "
              "site-for-site rather than only in shape. The chat-template sha "
              "e10ca381… is identical to the banked 8B hub AND to row 11, so the native "
              "arm is template-matched as well as lineage-matched: a native-arm "
              "difference cannot be a templating difference. "
              "IDENTITY (rake M9, verified independently of the dirname): config.json "
              "sha 95ef9768… is BYTE-IDENTICAL to the hub's "
              "meta-llama/Llama-3.3-70B-Instruct at hub revision 6f6073b4…; "
              "generation_config (sha 2fff3b8b…), model.safetensors.index.json and "
              "tokenizer_config.json are byte-identical to the hub as well. "
              "eos_token_id [128001, 128008, 128009] with generation_config "
              "temperature 0.6 / top_p 0.9 is the Llama-3 INSTRUCT signature (the base "
              "checkpoints carry the bare eos and no sampling defaults); tokenizer "
              "eos_token_id 128009 (<|eot_id|>). Template presence is recorded and is "
              "NOT a discriminator. "
              "WEIGHTS: 30 shards over 723 tensors, index-complete, 131.42 GiB bf16, "
              "loaded READ-ONLY from the shared weight store. The upstream repo's "
              "original/*.pth consolidated checkpoints were excluded by the trimmed "
              "pull — they are redundant with the safetensors set the collector reads, "
              "and the index verifies complete without them. "
              "SINGLE-CARD, BUT ONLY ON AN EMPTY CARD: 131.42 GiB against the job "
              "preflight's weights x1.25 allowance needs ~164 GiB, which fits a free "
              "card with roughly 14 GiB to spare and does NOT fit beside a substantial "
              "foreign resident — which is exactly why row 11 was collected through the "
              "sharded path. The fit is MEASURED by the job preflight at fire time, "
              "never assumed; if the card is occupied the job blocks rather than "
              "spilling. NEVER --allow-offload (standing desk ruling). "
              "CORPUS: all 780 frozen-corpus texts template on BOTH arms; longest "
              "sequence 1198 tokens native / 1095 raw against capacity 131072, so the "
              "position ceiling is nowhere near binding. Raw-arm specials prefix is "
              "[128000] (<|begin_of_text|>)."),
    RosterNode(
        key="mixtral-8x7b-instruct-v0.1",
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        roster_row=19, arms=("native", "raw"), num_hidden_layers=32, hidden_size=4096,
        weights_dirname="Mixtral-8x7B-Instruct-v0.1", checkpoint_identity="instruct",
        config_sha256="9d56d04b36d0fd12ff54ae4c5bac769cc176e254e64ff71144614b6318b40793",
        max_position_embeddings=32768,
        notes="THE THIRD MoE (external legibility): MixtralForCausalLM, 8 local experts, "
              "top-2 routing (num_experts_per_tok=2, num_local_experts=8), every layer "
              "an MoE layer, intermediate_size 14336. ~46.7B total / ~12.9B active — "
              "86.99 GiB bf16, comfortably single-card. Its value is that it sits at the "
              "OPPOSITE END of the sparsity design space from row 18's Qwen3-30B-A3B "
              "(128 experts, top-8, moe_intermediate_size 768) at a similar order of "
              "active parameters: few-wide-experts vs many-narrow-experts. With the "
              "carried DSV2-Lite that makes three MoE points spanning the design axis, "
              "so the MoE-vs-dense read is a panel rather than an anecdote. "
              "At 32 layers and hidden_size 4096 it shares BOTH its computed scan grid "
              "and its residual width with olmo2-7b-instruct and mistral-7b-instruct-"
              "v0.3, so the sparse rung can be read site-for-site against dense rungs of "
              "the same width — including its own family-mate. "
              "IDENTITY (rake M9, verified independently of the dirname): config.json "
              "sha 9d56d04b… is BYTE-IDENTICAL to the hub's "
              "mistralai/Mixtral-8x7B-Instruct-v0.1 at hub revision eba92302…; "
              "generation_config (sha 40e6ecbc…), model.safetensors.index.json and "
              "tokenizer_config.json are byte-identical to the hub as well. "
              "M9 FINDING, AND IT MATTERS: on this lineage the config sha is NOT a "
              "base-vs-instruct discriminator. mistralai/Mixtral-8x7B-v0.1 (the BASE "
              "repo) ships a config.json that is BYTE-IDENTICAL to the instruct repo's "
              "— same 9d56d04b… — and an identical generation_config (40e6ecbc…, a bare "
              "`_from_model_config` stub with eos 2 / bos 1 and no sampling defaults, so "
              "there is no temperature/top_p signature to check either). Even the "
              "shard sizes agree. A config-sha check alone therefore proves only "
              "'some Mixtral-8x7B-v0.1-lineage checkpoint', never WHICH ONE, and the "
              "usual Llama/Qwen discriminator set silently degenerates here. What DOES "
              "separate them: tokenizer_config.json — VALUE OF RECORD 47536143… on the "
              "instruct repo (this checkpoint) against 747ec9… on the base repo — and, "
              "inside it, the chat template: the base repo carries NONE, the instruct "
              "repo carries the 1058-char v0.1 [INST] template, sha "
              "796853172235ab3aaa4810013ddfe2eab481bda3306dad4f700c00d436eed596 "
              "(itself distinct from the Mistral-7B-v0.3 template row 13 uses), "
              "reproduced from this node's own collection stamp. This is "
              "the ONE rung where template presence is load-bearing rather than merely "
              "recorded, which is the exact inverse of the Qwen3 case, so the general "
              "rake M9 rule ('presence proves nothing') must be applied per-lineage and "
              "not as a reflex. Weights-level backstop, the discriminator of last "
              "resort: the first shard's sha256 is 54669c5a… on the instruct repo vs "
              "b43400ce… on the base repo, and the checkpoint of record matches the "
              "former. All five fields are valued on BOTH sides in "
              "`metabasis.lineage_discriminators`, which CHECKS that each one claimed "
              "here actually separates the pair (rake M17(a)) — run it, do not re-read "
              "this paragraph. "
              "WEIGHTS: 19 shards over 995 tensors, index-complete, loaded READ-ONLY "
              "from the shared weight store. The upstream repo's consolidated.*.pt "
              "torch checkpoints were excluded by the trimmed pull — redundant with the "
              "safetensors set, and the index verifies complete without them. "
              "CORPUS: all 780 frozen-corpus texts template on BOTH arms — VERIFIED, "
              "not assumed, because the older Mistral-lineage templates reject a "
              "standalone system turn and this node's template is the v0.1 one, not the "
              "v0.3 one that row 13 relies on. Longest sequence 1679 tokens native / "
              "1612 raw against capacity 32768. Raw-arm specials prefix is [1] (<s>), "
              "the same as row 13."),
)

# The big-chain MULTICARD rungs (2026-07-28): roster rows 20 and 21, the two
# members of the big-chain pull that exceed one card and therefore collect
# through the certified sharded path (prereg §4; ledger collection/shard-cert,
# where a forced 8-way sharded 8B collection came out byte-identical to a fresh
# single-device one). Architecture facts read from each checkpoint's own
# config.json on the shared weight store; identity verified per rakes M9 AND
# M17 — the discriminator is chosen per lineage and SHOWN to separate the
# confusable siblings before it is trusted.
#
# Both nodes are sharded-only by arithmetic, not by preference: 755.96 GiB
# (row 20) and 1249.88 GiB (row 21) of bf16 weights against a 179.06 GiB card.
BIG_CHAIN_MULTICARD: tuple[RosterNode, ...] = (
    RosterNode(
        key="llama-3.1-405b-instruct", model_id="meta-llama/Llama-3.1-405B-Instruct",
        roster_row=20, arms=("native", "raw"), num_hidden_layers=126, hidden_size=16384,
        weights_dirname="Llama-3.1-405B-Instruct", checkpoint_identity="instruct",
        config_sha256="a55a4fc4b5b6194a1571f435bdc15fe61fe59eddf4c28c1540a79eb89596d345",
        max_position_embeddings=131072,
        notes="THE DENSE SCALE CEILING. Row 20 completes the lineage-matched ladder "
              "3B -> 8B -> 70B (rows 11/12) -> 405B: two orders of magnitude inside ONE "
              "pretrain family, which is what makes the scale read a ladder rather than "
              "a scatter. It is also the prereg's SCALE-PROBE AUDIT HUB (§3 quad-hub): "
              "the direct test of the larger-hubs-transfer-better conjecture, so its "
              "constants are re-derived for the audit set once its fits are banked. "
              "LlamaForCausalLM, 126 layers, hidden_size 16384 (the widest residual "
              "stream on the roster by 2x), 128 heads, intermediate 53248, vocab 128256, "
              "the same llama3 rope_scaling (factor 8.0, original max_position_embeddings "
              "8192) as rows 11/12 and the banked 8B hub. 126 layers is NOT 80, so unlike "
              "the 11-vs-12 pair the scan grid differs from the 70B rungs' and the "
              "comparison is by fractional depth, not site-for-site. "
              "IDENTITY (rake M9, verified independently of the dirname): config.json "
              "sha a55a4fc4... is BYTE-IDENTICAL to the hub's "
              "meta-llama/Llama-3.1-405B-Instruct; generation_config (sha ececd938...), "
              "model.safetensors.index.json and tokenizer_config.json are byte-identical "
              "to the hub as well. eos_token_id [128001, 128008, 128009] with "
              "generation_config temperature 0.6 / top_p 0.9 is the Llama-3 INSTRUCT "
              "signature; tokenizer eos_token_id 128009 (<|eot_id|>); chat-template sha "
              "e10ca381... is IDENTICAL to the banked 8B hub and to rows 11/12, so the "
              "native arm is template-matched as well as lineage-matched. Weights-level "
              "backstop (rake M17's universal discriminator): the first shard's sha256 is "
              "04160c8e... and matches the hub's LFS digest. "
              "WEIGHTS: 191 shards over 1137 tensors, index-complete, 755.96 GiB bf16 "
              "(405.85B params), loaded READ-ONLY from the shared weight store. "
              "SHARDED, 8 cards: 748.13 GiB of that is the 126 decoder layers at a "
              "UNIFORM 5.94 GiB each, so the certified even-layer split "
              "(--shard-across 8 --assert-multi-device) lands at most 102.8 GiB on any "
              "card (card 0, which also carries the 7.83 GiB of embeddings/head/norm) "
              "against a 179.06 GiB card — 76 GiB of headroom. NEVER --allow-offload "
              "(standing desk ruling). "
              "CORPUS: capacity 131072 against a corpus whose longest sequence on the "
              "shared Llama-3 tokenizer is ~1200 tokens, so the position ceiling is "
              "nowhere near binding. Raw-arm specials prefix is [128000] "
              "(<|begin_of_text|>), as on every Llama-3 rung."),
    RosterNode(
        key="dsv3", model_id="deepseek-ai/DeepSeek-V3",
        roster_row=21, arms=("native", "raw"), num_hidden_layers=61, hidden_size=7168,
        weights_dirname="DeepSeek-V3", checkpoint_identity="instruct",
        config_sha256="cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9",
        max_position_embeddings=163840,
        notes="THE FRONTIER MoE (671B, cross-lab), and the fourth MoE point on the "
              "roster beside the carried DSV2-Lite, row 18's Qwen3-30B-A3B and row 19's "
              "Mixtral. DeepseekV3ForCausalLM, 61 decoder layers, hidden_size 7168, "
              "256 routed experts + 1 shared, top-8 routing, first_k_dense_replace=3 "
              "(layers 0-2 are dense MLP, layers 3-60 are MoE), moe_intermediate_size "
              "2048, MLA attention (q_lora_rank 1536, kv_lora_rank 512, qk_rope_head_dim "
              "64), yarn rope. The checkpoint also ships an MTP layer 61 "
              "(num_nextn_predict_layers=1) which transformers does not build — the model "
              "is 61 layers, indices 0..60, and the scan grid tops out at L52. "
              "DTYPE REGIME — prereg ADDENDUM 2026-07-26-A, BINDING: the checkpoint is "
              "native FP8 (e4m3, block [128,128], 641.30 GiB on disk), but row 21's named "
              "FP8 deviation is RETAINED AS FALLBACK ONLY. DSV3 collects and builds in "
              "bf16 via dequantize-on-load, FineGrainedFP8Config(dequantize=True) — the "
              "standard roster regime, standard bitwise spot-replay gate, standard "
              "differentiable input-gradient path. THE ADDENDUM'S NAMED MECHANISM DOES "
              "NOT WORK ON THIS CHECKPOINT and the reason is structural, measured "
              "2026-07-28: self_attn.kv_a_proj_with_mqa.weight is [576, 7168] — 576 = "
              "kv_lora_rank 512 + qk_rope_head_dim 64 — and the checkpoint stores its "
              "block scales as [5, 56], i.e. ceil(576/128) rows with a PARTIAL last "
              "block, while the pinned transformers' Fp8Dequantize hard-raises on any "
              "shape that is not a whole number of 128-blocks. The tensor occurs ONCE "
              "PER LAYER (62 in the index), so the first layer kills the load — "
              "identically for a 5-layer slice and for the full model. The REGIME is "
              "reached instead by dequantizing AHEAD OF TIME into a bf16 mirror whose "
              "arithmetic is bitwise-checked against transformers' own Fp8Dequantize on "
              "every block-divisible tensor (122/122 exact on the first shard at the "
              "pre-fire check); the collector then loads an ordinary bf16 checkpoint "
              "with no quantization_config and no --dequantize-fp8 at all. THE "
              "MECHANISM SUBSTITUTION IS A DESK/LUXIA MATTER — the science the addendum "
              "ruled (bf16 forward, standard gate, differentiable target build) is "
              "unchanged, but the addendum names a means that is not executable here. "
              "The --dequantize-fp8 flag remains in the collector, correct and "
              "selftested, for any FP8 checkpoint whose blocks all divide evenly. "
              "IDENTITY — THE RAKE-M17 CASE IN ITS PUREST FORM. On this lineage the "
              "ENTIRE metadata surface degenerates: deepseek-ai/DeepSeek-V3 (chat) and "
              "deepseek-ai/DeepSeek-V3-Base ship a BYTE-IDENTICAL config.json (both "
              "cbf0b95d...), a BYTE-IDENTICAL tokenizer_config.json (both 637bcd1a...), "
              "hence the identical chat template (3b8267e5..., so template presence is "
              "not merely uninformative here but actively misleading), and NEITHER repo "
              "ships a generation_config.json at all — there is no eos/sampling signature "
              "to check. Config sha, template sha, tokenizer sha and eos ids ALL fail to "
              "discriminate. The discriminator of record is therefore the WEIGHTS "
              "themselves (M17 rule (b), the universal fallback): the first shard's "
              "sha256 is b933b099... on the checkpoint of record and matches "
              "deepseek-ai/DeepSeek-V3's hub LFS digest, while the BASE checkpoint's "
              "first shard hashes 3f4e5fce... — SHOWN to separate them (M17 rule (a)), "
              "not merely assumed to. DeepSeek-V3.1 is separately "
              "refuted by config (sha 3e5d192d..., and its quantization_config carries "
              "scale_fmt 'ue8m0', which the pinned transformers does not know — the "
              "scales would be mis-read). The row-21 requirement of native+raw arms is "
              "what makes the CHAT checkpoint the only admissible one (the "
              "arm-consistency rule), and the weights check is what proves we have it. "
              "WEIGHTS: 163 shards over 91991 tensors (45808 of them weight_scale_inv), "
              "index-complete, loaded READ-ONLY from the shared weight store. The index's "
              "total_size (1275.04 GiB) is a bf16-ASSUMED figure — it counts 2 bytes per "
              "element regardless of dtype — and must never be quoted as a disk size. "
              "SHARDED, 8 cards, and this is the tightest node on the roster BY DESIGN: "
              "bf16-resident is 1249.88 GiB (671.03B params) = 87.2% of the 8-card node. "
              "The layers are HETEROGENEOUS — 3 dense at 1.09 GiB and 58 MoE at 21.43 GiB "
              "— and 58 MoE layers over 8 cards forces at least two cards to hold 8 of "
              "them, so 8 x 21.43 = 171.5 GiB is the arithmetic FLOOR for any map that "
              "keeps each decoder layer whole (which pipeline-parallel device_map does, "
              "and which is what the sharding certification covers). The certified even "
              "split (--shard-across 8) realizes exactly that floor: per-card "
              "[113.9, 171.5, 150.0, 171.5, 171.5, 150.0, 171.5, 150.0] GiB against a "
              "179.06 GiB card, i.e. 7.6 GiB of headroom on the worst card, and no "
              "byte-balanced alternative does better. The job preflight MEASURES free "
              "memory and the load transient rather than assuming this. NEVER "
              "--allow-offload (standing desk ruling). "
              "ARMS: capacity 163840, so the position ceiling is not binding. Per the "
              "FP8 lane design the raw arm carries NO specials prefix (p=0) — the "
              "tokenizer_config's add_bos_token is discarded by the pinned transformers "
              "when tokenizer.json exists — which the carried DSV2-Lite anchor shares. "
              "The native template has no strftime_now, so build_ids' date_string is "
              "accepted-and-ignored rather than raising; the branch taken is recorded."),
)

# The CARRIED banked nodes (2026-07-29). Prereg roster rows 1-6 (3b, 8b,
# qwen-7b, dsv2-lite, gemma-3-27b, olmo2-7b) were collected and banked before
# this campaign's registries existed. They are NOT new collections — every one
# is already ratified in `fit_transport_maps.SITES` — so a row here does
# something different from every group above: it does not make the node
# collectable, it gives an already-banked node the M9 DRIFT-DETECTION GUARD (a
# config sha anchor) that every other roster node carries and these had none of.
# A carried node therefore enters ROSTER ALREADY GRADUATED, which is why its row
# number sits below wave-1's and its scan grid is the grid the scan ceremony
# actually ran rather than a grid still to be run.
#
# ONLY ROW 5 IS HERE SO FAR. The other five carried nodes still have no roster
# row and therefore still no config-sha anchor; that is a NAMED GAP, not an
# omission this tuple's shape implies is closed. They join as their anchors are
# verified, one row at a time, on the evidence each one's own audit produces.
#
# Adding a key here has one mechanical consequence worth stating: SCAN_GRIDS
# gains the key, so `sites_for`/`scan_grid_table`/`collectable` start reporting
# it. It changes NO site registration — `fit_transport_maps.SITES` and
# `read_composed_predictions.SITE_OF_RECORD` are set from RULINGS, never from a
# roster row, and this row moved neither. gemma3-27b's carried-provisional L36
# was subsequently RETIRED by Luxia's site ruling (2026-07-29): the registered
# grid is now (38, 41), ⋆ L38, and the ruling is recorded where the
# registrations live, not here (see the interleave note below for the
# architecture facts the ruling consumed).
CARRIED_BANKED: tuple[RosterNode, ...] = (
    RosterNode(
        key="gemma3-27b", model_id="google/gemma-3-27b-it",
        roster_row=5, arms=("native", "raw"), num_hidden_layers=62,
        hidden_size=5376,
        weights_dirname="models--google--gemma-3-27b-it",
        checkpoint_identity="instruct",
        config_sha256="cabd884f5e0d4f01a5bd7fe14bd4bacd0bd83f3725a02acbdc4e72dc001835fa",
        # max_position_embeddings: DELIBERATELY UNSET — see the notes. The
        # checkpoint's config.json DECLARES NO max_position_embeddings at all;
        # None here means ABSENT-FROM-CONFIG, verified, not un-read.
        scan_grid_extension=(34, 36, 38),
        notes="THE FIRST CARRIED BANKED NODE TO GET A ROSTER ROW (prereg row 5, "
              "'anchor, 3rd family (carried; banked)'). Architecture facts read "
              "from the checkpoint's own config.json 2026-07-29 at the PINNED "
              "revision 005ad3404e59d6023443cb575daa05336842228a: 62 layers, "
              "hidden_size 5376, 32 attention heads, 16 kv heads (GQA 2:1), "
              "head_dim 128 (declared, and NOT hidden_size/heads — 5376/32 = 168, "
              "so head_dim must be read, never computed, on this lineage), eos "
              "[1, 106], architecture `Gemma3ForConditionalGeneration`. That "
              "architecture is the MULTIMODAL WRAPPER: the decoder stack resolves "
              "through the `language_model` branch (hooks.decoder_layers handles "
              "it; `metabasis.config.MODEL_PRESETS['gemma3-27b']` carries the same "
              "facts for the probe side). "
              "⚠ max_position_embeddings IS ABSENT FROM THE CONFIG — the field "
              "above is None because the checkpoint DECLARES NONE, not because "
              "nobody looked. Recorded explicitly so no reader ever back-fills a "
              "plausible default (8192/32768/131072 are all wrong here: the config "
              "says nothing, and a job preflight's corpus-length check must "
              "therefore be satisfied some other way, or the gap ruled on). This "
              "is the ONE roster node whose positional capacity is not a number. "
              "IDENTITY (rakes M9/M17) — THIS LINEAGE INVERTS MIXTRAL, so the "
              "discriminator is CHOSEN and SHOWN rather than reused: the "
              "DISCRIMINATOR OF RECORD IS THE CONFIG SHA. config.json hashes "
              "cabd884f… on `-it` (this checkpoint) against 019693e9… on the "
              "sibling `-pt` BASE repo — they separate. generation_config.json is "
              "BYTE-IDENTICAL between it and pt and therefore proves NOTHING here, "
              "which is the exact inverse of the Llama/Qwen rows where the "
              "sampling defaults are the signature; tokenizer_config.json "
              "separates them too and is recorded as the corroborating check, not "
              "the discriminator of record. WEIGHTS-LEVEL BACKSTOP (M17 rule (b), "
              "and here it is not the last resort but an independent pass): 12/12 "
              "shards verified BITWISE against the hub's LFS sha256 at the pinned "
              "revision, 54.86 GB, untouched on the shared store since 2026-07-12 "
              "— M9's mutation threat is excluded at the weights level, not "
              "inferred from metadata. "
              "RULED GRID EXTENSION (34, 36, 38) — the deviation, recorded here so "
              "the extension history is machine-readable and `scan_grid` stays "
              "re-derivable from num_hidden_layers alone. The computed 12-site "
              "[0.15,0.85] grid is (9,13,17,21,25,29,33,37,41,45,49,53) and does "
              "NOT contain 36; the WH6-stamped banked peak region {34,36,38} is "
              "where the ENTIRE banked gemma object roster lives, so the scan "
              "ceremony ran the effective 15-site grid (the additions were DERIVED "
              "in-preflight from scan_grid(62) and re-added, never typed). This "
              "tuple is SCAN HISTORY — what the ceremony ran — and is deliberately "
              "NOT moved by a site ruling; +35 is likewise NOT in it (the "
              "site-evidence pass since ran a 16-site re-collect that measured "
              "L35, reproducing all 15 original sites bitwise, so adding it is a "
              "separate ruling on this field, not a consequence of the site one). "
              "THE REGISTERED FIT GRID IS NOW `fit_transport_maps.SITES"
              "['gemma3-27b']` = (38, 41), ⋆ L38 — RULED BY LUXIA 2026-07-29 from "
              "the six-site â evidence table (readout `7f59af50…`), retiring the "
              "carried-provisional L36 to scanned-history; L38 and L41 are both on "
              "the effective grid (41 is in the computed 12), so the ratification "
              "invariant is satisfied by construction. "
              "ATTENTION INTERLEAVE, A NAMED ARCHITECTURE FACT THE SITE MACHINERY "
              "MAY CONSULT: gemma-3 is 5:1 LOCAL:GLOBAL interleaved — sliding "
              "window 1024, and layer i is GLOBAL iff (i+1) % 6 == 0, i.e. "
              "{5,11,17,23,29,35,41,47,53,59}. Consequences that are material to "
              "the 2026-07-29 site ruling and must not be rediscovered: the ⋆ site "
              "L38 IS A LOCAL sliding-window layer, and its 1024-token window is "
              "MARGINALLY BINDING against the longest templated corpus text (1134 "
              "tokens) — as is true of every one of the three ruled scan additions "
              "(34, 36, 38); the robustness site L41 IS GLOBAL, so the ruled pair "
              "straddles the interleave BY ACCIDENT, not by design. L35 (L36's "
              "global neighbour, UNMEASURED when this row was written) has since "
              "been measured by the site-evidence pass, and the answer is that "
              "ATTENTION TYPE IS A NON-VARIABLE HERE: at matched depth L35-vs-L36 "
              "differ by ≤.007 in â with the two hub columns disagreeing on the "
              "sign, and the site ordering tracks depth/coherence instead. Of the "
              "effective 15-site grid only {17, 29, 41, 53} are global. "
              "⚠ THE SLIDING WINDOW IS NOT A POSITION CEILING and must never be "
              "written as one: `max_seq_len` stays unset. Truncating the corpus to "
              "1024 would change the object every other node is compared against, "
              "whereas a sliding window is a per-layer attention span the model "
              "applies to the FULL sequence. The prereg ADDENDUM 2026-07-27-B "
              "truncation is for LEARNED-absolute-position ceilings (gpt2-xl); it "
              "does not apply here. "
              "WEIGHTS LAYOUT DEVIATION: unlike every node above, this checkpoint "
              "lives in an HF SNAPSHOT CACHE, not the flat weights root — "
              "`weights_dirname` is the cache's repo directory basename and the "
              "actual weights are one level further down under "
              "`snapshots/<revision>` at the pinned revision above. The full path "
              "is passed via --model-path as always and is never hardcoded here; "
              "the carried banked-4 nodes (qwen-7b, dsv2-lite) share this layout, "
              "which is why a flat-root sweep does not find them (rake M35)."),
)

# The BEHAVIORAL TIER (2026-07-29): prereg roster row 22, and the first node on
# the roster whose reason for existing is the TRANSPORTED-WRITE column rather
# than the fit/star column. The tier is a ROLE, and the role is narrower than
# every group above it: row 22 is a write TARGET only — NO native
# entropy-gradient target is built for it, and it carries NO star row. It
# therefore never acquires a site of record, never enters
# `read_composed_predictions.SITE_OF_RECORD`, and never appears in the candidate
# enumeration; a row here buys exactly one thing, which is the same one thing a
# row buys anywhere in this module — SCAN_GRIDS gains the key, so
# `collect_mean_states.py` will accept it (MODEL_KEYS = SITES ∪ SCAN_GRIDS).
#
# `RosterNode` HAS NO TIER FIELD and this group does not add one: the roster
# models role distinctions with `arms`, `roster_row` and named deviations, and
# states the rest in the row's own notes (row 16's "raw arm only, per prereg row
# 16" is the precedent). The tier is therefore recorded in the notes, first line
# and in the prereg's words, plus this header — not as a new column that only
# one row would ever value.
#
# The other reason row 22 exists: falcon-mamba-7b-instruct is the FIRST
# NON-TRANSFORMER on the roster — a pure state-space model, no attention layer
# anywhere in the stack. Every other node's residual stream is produced with
# attention in the loop; this one's is produced by a recurrence. A transported
# write that lands here is evidence that the axis being transported is not an
# artifact of the attention stack.
#
# zamba2-7b — the other SSM-lineage candidate from the same pull — is
# DELIBERATELY ABSENT. It is hard-parked on an upstream transformers defect and
# its row waits on Luxia's shim ruling. Adding a key here is what makes a node
# collectable, so the absence IS the block (the BIG_CHAIN_MULTICARD precedent,
# applied to a defect instead of to a scan design).
#
# Architecture, identity and corpus facts below are the SSM TOOLING GATE of
# 2026-07-29 — measured on the checkpoint, not re-derived here.
BEHAVIORAL_TIER: tuple[RosterNode, ...] = (
    RosterNode(
        key="falcon-mamba-7b-instruct",
        model_id="tiiuae/falcon-mamba-7b-instruct",
        roster_row=22, arms=("native", "raw"), num_hidden_layers=64,
        hidden_size=4096,
        weights_dirname="falcon-mamba-7b-instruct",
        checkpoint_identity="instruct",
        config_sha256="b588a09876a43945bbfb56aacfeadb23985def98925efa1a0d072ccdd0bc54c1",
        # max_position_embeddings: DELIBERATELY UNSET — see the notes. A pure SSM
        # HAS no positional ceiling to record (there is no position table and no
        # attention window), and the config declares no `max_position_embeddings`
        # at all. None here means ABSENT-VERIFIED, not un-read — the gemma3-27b
        # precedent for an absent field, reached for a different reason.
        notes="BEHAVIORAL TIER (frozen prereg roster row 22): TRANSPORTED-WRITE "
              "COLUMN ONLY. No native entropy-gradient target is built for this "
              "node and it carries NO star row, so it has no site of record and "
              "no candidate slot — it is written INTO, never fitted FROM as a "
              "star member. Everything below is the SSM TOOLING GATE of "
              "2026-07-29, measured on the checkpoint. "
              "ARCHITECTURE — THE FIRST NON-TRANSFORMER ON THE ROSTER: "
              "FalconMambaForCausalLM, a PURE SSM with no attention layer at "
              "all; 64 FalconMambaBlock layers in the container "
              "`backbone.layers`, hidden_size 4096. The stack resolves through "
              "`hooks.decoder_layers`'s base_model_prefix FALLBACK path "
              "(base_model_prefix='backbone'; the two original checks miss "
              "because this lineage neither nests under `.model` nor exposes "
              "`.layers` on the LM head class) — the same fallback that already "
              "covers GPTNeoX's `.gpt_neox.layers` and GPT-2's `.transformer.h`, "
              "reached here for a third lineage without changing what it returns "
              "for any of them. Site L keeps the campaign's meaning unchanged: a "
              "forward_pre_hook on `backbone.layers[L]`, i.e. the residual "
              "ENTERING block L, valid range [0, 63]. "
              "⚠ NAMED LINEAGE FACT — RAKE M37, AND IT IS A TRAP FOR ANYONE "
              "SPOT-CHECKING THIS NODE: on the Mamba lineage "
              "`output_hidden_states` is OFF BY ONE against the transformer "
              "convention — hidden_states[i] is block i's OUTPUT, not its input. "
              "The collector's hook convention is the CORRECT one and was "
              "verified BITWISE 64/64 at the SHIFTED reference; a check that "
              "compares site L's capture against hidden_states[L] will disagree "
              "on every layer and the capture is not what is wrong. "
              "⚠ max_position_embeddings IS ABSENT FROM THE CONFIG, and unlike "
              "gemma3-27b the absence is not a gap to be ruled on — a pure SSM "
              "carries its context in a recurrent state, so there is NO "
              "positional ceiling to declare and none is missing. Recorded as "
              "None = absent-verified so no reader back-fills a plausible "
              "number. NO TRUNCATION DEVIATION IS NEEDED OR TAKEN: the frozen "
              "corpus fits AS WRITTEN on both arms and templates 780/780 clean "
              "on both, so `max_seq_len` stays unset and this node compares "
              "against the same frozen text objects as every other node (prereg "
              "ADDENDUM 2026-07-27-B is for LEARNED-absolute-position ceilings "
              "like gpt2-xl's, and does not reach here). "
              "IDENTITY (rakes M9/M17, verified independently of the dirname) — "
              "THE DISCRIMINATOR OF RECORD IS THE CONFIG SHA, and it is SHOWN to "
              "separate rather than assumed to: config.json hashes b588a098… on "
              "`-instruct` (this checkpoint) against 08ec4cda… on the sibling "
              "BASE repo tiiuae/falcon-mamba-7b — they separate. CORROBORATORS, "
              "recorded as the second and third independent passes, not as the "
              "discriminator: generation_config eos [11, 10] here vs 11 on base, "
              "bos 8 vs 0, pad 0 vs 11; chat-template sha "
              "a805e50fed68938a076b07e2e602639611b50b1ced0e50f11eb92f1ba25be4dc "
              "(template presence itself proves nothing — rake M9 — but the "
              "differing eos/bos/pad triple does). "
              "⚠ THE CONFIG'S OWN `_name_or_path` SAYS 'falcon-mamba-7b-chat' — "
              "a THIRD name, matching neither the repo id nor the directory. It "
              "is upstream's artifact and is present in the hub file too, so it "
              "is not evidence of a swapped or mutated checkpoint; it is simply "
              "a name, and names are never trusted here (M9). Recorded because a "
              "later reader WILL find it and must not spend the discovery twice. "
              "SCAN GRID: 64 layers, so the computed 12-site [0.15,0.85] grid is "
              "(10,14,18,22,26,30,34,38,42,46,50,54) — DERIVED by `scan_grid`, "
              "never typed here — which is the SAME grid as qwen2.5-32b-instruct "
              "(row 10, also 64 layers), so the SSM can be read site-for-site "
              "against a dense transformer of equal depth rather than only by "
              "fractional depth. No ruled extension. hidden_size 4096 is shared "
              "with olmo2-7b-instruct, mistral-7b-instruct-v0.3 and pythia-6.9b, "
              "so the write target's residual width is not a new variable "
              "either. "
              "WEIGHTS: the flat weights root layout (a plain directory named by "
              "`weights_dirname`, not an HF snapshot cache), 3 shards over 643 "
              "tensors, index-complete. The full path is passed via --model-path "
              "as always and is never hardcoded here."),
)

# The webtext-v3 SCALE ADD (2026-08-03): roster row 24, the third checkpoint the
# frozen webtext-v3 prereg §5 names in the CORE roster ("the 17 collected star
# nodes + Qwen2.5-72B-Instruct + Llama-3.1-8B base + Qwen2.5-7B base") and the
# only one of the three that is an INSTRUCT checkpoint. It exists for the §7 C2
# READ (recipe-vs-range): this node's descriptive hub rank and median |e| against
# two PRE-NAMED outcomes — top-tier-equivalent at 72B supports the recipe
# reading, degradation to llama-3.1-70b-instruct's tier supports the
# range-artifact reading — scored by a paired sign test against row 11 on their
# shared slots. The row is therefore only useful if it is comparable to the
# 70Bs, which is what the grid note below is about.
#
# ⚠ BANK KEY, AND IT IS A CHOICE THIS ROW HAD TO MAKE: the frozen §5/§7 prose
# names this checkpoint "Qwen2.5-72B-Instruct" and "qwen2.5-72b" and registers no
# bank key. `qwen2.5-72b-instruct` continues the family's OWN key convention on
# this roster (rows 8/9/10: qwen2.5-3b-instruct · qwen2.5-14b-instruct ·
# qwen2.5-32b-instruct) rather than the prose's short form. Nothing was collected
# under any other spelling — the bank does not exist yet — so this is a naming
# decision and not a rename.
#
# ARCHITECTURE + IDENTITY are read from the staged, sha-certified pull of
# 2026-08-03 (ledger "THE ROSTER-V3 WEIGHTS ARE STAGED"), never from this file's
# memory: the checkpoint's own config.json at the pinned revision, and the
# node-side `.metabasis-provenance.json` the fetcher wrote beside it.
V3_SCALE_ADD: tuple[RosterNode, ...] = (
    RosterNode(
        key="qwen2.5-72b-instruct", model_id="Qwen/Qwen2.5-72B-Instruct",
        roster_row=24, arms=("native", "raw"), num_hidden_layers=80,
        hidden_size=8192, weights_dirname="Qwen2.5-72B-Instruct",
        checkpoint_identity="instruct",
        config_sha256="14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29",
        max_position_embeddings=32768,
        notes="WEBTEXT-V3 CORE ADD (frozen prereg §5), and the §7 C2 "
              "recipe-vs-range node: the Qwen2.5 family's TOP RUNG, set against "
              "llama-3.1-70b-instruct (row 11) by a paired sign test on their "
              "shared slots. PINNED REVISION "
              "495f39366efef23836d0cfae4fbe635880d2be31 — the staging pull of "
              "2026-08-03 fetched that commit (47 files, 145.42 GB) and verified "
              "every LFS file's sha256 against the hub at it, two-sided, 100% of "
              "weight bytes; the node-side provenance record carries the same "
              "revision as `head_at_fetch`, and a COLD spot-rehash reproduced "
              "config.json 14ca2173…, the first and last shards (18d5d2b7… / "
              "d9e72766…) and the index (6c1d85ca…). Nothing was excluded from "
              "this repo. "
              "ARCHITECTURE, read from the checkpoint's own config.json at that "
              "revision: Qwen2ForCausalLM, 80 decoder layers, hidden_size 8192, "
              "64 attention heads / 8 kv heads (GQA 8:1), intermediate_size "
              "29568, vocab 152064, rope_theta 1e6, tie_word_embeddings False, "
              "torch_dtype bfloat16, transformers 4.43.1. "
              "POSITIONAL CAPACITY 32768 — recorded because it is the SMALLEST "
              "on the Qwen2.5 rungs (rows 8/9/10 and the base siblings all carry "
              "131072) and because the job preflight's corpus-length check is "
              "mechanical rather than folklore. `sliding_window` 131072 appears "
              "in the config with `use_sliding_window` FALSE, so it is inert and "
              "32768 is the ceiling that binds. No `max_seq_len` is set: the "
              "corpus fits or the preflight blocks, and a truncation is a ruling, "
              "not a default. "
              "IDENTITY (rakes M9/M17(a)), STATED WITH ITS LIMIT: the "
              "DISCRIMINATOR OF RECORD IS THE CONFIG SHA 14ca2173…, and what it "
              "is verified against is the HUB — the two-sided pull above compares "
              "this tree file-for-file with Qwen/Qwen2.5-72B-Instruct at the "
              "pinned revision, so the row does not rest on the directory name "
              "(M9). ⚠ NO SAME-SIZE SEPARATION WAS PERFORMED, because there is "
              "nothing to separate from: Qwen2.5-72B (the base repo) is NOT on "
              "either store, so unlike rows 25/26 this row shows no "
              "base-vs-instruct discriminator table for its OWN size. What IS "
              "shown, from the same staged config log: eos_token_id 151645 here "
              "against 151643 on the staged Qwen2.5-7B BASE config (bos is "
              "151643 on both) — the family's instruct signature, read off two "
              "configs in one log rather than assumed. The conventional NAMES "
              "for those ids (<|im_end|> / <|endoftext|>) are not re-verified "
              "here; the IDS are what the log holds and what this row claims. "
              "CHAT TEMPLATE: tokenizer_config.json sha "
              "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 "
              "(7305 bytes), which MATCHES the value row 26 records for the "
              "already-banked Qwen2.5-7B-Instruct sibling (`5b5d4f65…`) — the "
              "Qwen2.5 instruct rungs appear to ship ONE template, so the native "
              "arm here is template-matched to the banked `qwen-7b`. Recorded as "
              "an observation at the recorded (prefix) precision, NOT as an arm "
              "proof: the arm proof is `build_ids` at collection preflight, which "
              "is where every other row's template facts came from. "
              "ARMS native+raw by the instruct-class rule (frozen §3.2 scores an "
              "instruct endpoint in both arms), the same rule rows 8/9/10 run "
              "under — not an inference from the template sha above. "
              "SCAN GRID: 80 layers, so the computed 12-site [0.15,0.85] grid is "
              "(12,17,22,27,32,37,43,48,53,58,63,68) — DERIVED by `scan_grid`, "
              "never typed. IT IS THE SAME GRID AS ROWS 11 AND 12 "
              "(llama-3.1-70b-instruct and llama-3.3-70b-instruct, also 80 "
              "layers), which is what makes the C2 comparison readable "
              "SITE-FOR-SITE and not only by fractional depth. No ruled "
              "extension. "
              "SITES REGISTERED (58, 63) ⋆ L58 — RULED BY LUXIA 2026-08-04 "
              "(ledger \"SITE RULING\") on this node's own webtext-v3 12-site "
              "scan fits (cp2_summary `5f787171…`, 72/72 cells valid in BOTH "
              "arms, on the FROZEN --splits-artifact membership). The frozen §5 "
              "ceremony this row previously recorded as OWED has now run: L58 is "
              "the peak on FIVE OF THE SIX instruments (all three native fit "
              "families + both raw procrustes families + raw cka) and L63 is "
              "kept because the SIXTH — native cka — peaks there, on a broad "
              "L48–L63 plateau; the pair is the read. Raw ridge inflates the "
              "shallow half (the rows-25/26 divergence pattern with the roles "
              "swapped); the families agree from L27 on and nothing shallow is "
              "registered. `fit_transport_maps.SITES` and "
              "`read_composed_predictions.SITE_OF_RECORD` carry the same ruling "
              "and the full rationale; the two registries must agree. "
              "NO `scan_grid_extension` IS NEEDED, AND THAT IS THE DIFFERENCE "
              "FROM ROWS 25/26: both 58 and 63 are ON the computed 12-site grid "
              "above, so the ratification invariant is satisfied by construction "
              "— a curve-visited pair, nothing ruled off-grid, and no re-collect "
              "implied by the ruling. "
              "⚠ (58, 63) ⋆ L58 IS EXACTLY ROW 12's REGISTERED GRID "
              "(llama-3.3-70b-instruct), reached INDEPENDENTLY off this node's "
              "own scan — a coincidence worth naming because it makes the C2 "
              "footing against row 12 site-IDENTICAL as well as grid-identical. "
              "C2's own paired sign test is scored against ROW 11, which rules "
              "to (37, 43) ⋆ L37. The desk flagged a rider on precisely this: "
              "the 72B's r² TROUGH sits on row 11's sites and its r² PEAK on row "
              "12's, so a C2 comparison quoted at the other row's site reads this "
              "node at an extremum of its own curve. Site-for-site is a footing, "
              "not an immunity. "
              "COLLECTED (2026-08-03→04, the wave-3 tail), UNDER A SCHEDULING "
              "DEVIATION THAT WAS NEVER A `blocked_reason`: at 136 GiB on disk "
              "this node exceeds the 100 GiB cohabitation ceiling on two cards, "
              "so the desk ruled SHARD-ACROSS-3 for it and the 70B pair (72.2 "
              "GiB/card, a sharded job runs as the SOLE job) and it ran that way. "
              "That was a deviation the JOB carried, not a registry block, which "
              "is why `blocked_reason` stayed None throughout. "
              "WEIGHTS: the metabasis-owned v3 roster tree on the RAID (standing "
              "placement ruling 2026-07-29), a plain directory named by "
              "`weights_dirname`, 37 shards, index-complete; the full path is "
              "passed via --model-path and is never hardcoded here."),
)

# The webtext-v3 BASE SIBLINGS (2026-08-03): the two base checkpoints the frozen
# webtext-v3 prereg §5 names in the CORE roster ("the 17 collected star nodes +
# Qwen2.5-72B-Instruct + Llama-3.1-8B base + Qwen2.5-7B base"). They exist to make
# the §C4 base-vs-instruct sibling read possible: each is the pretrain parent of a
# checkpoint already on the roster, so the pair differs in post-training and in
# nothing else.
#
# ⚠ THE `roster_row` NUMBERS ARE REGISTRY BOOKKEEPING, NOT A CITATION, AND THE
# DESK MAY RENUMBER THEM WITH ONE EDIT. The numbered roster TABLE is the campaign
# prereg's (rows 1–23); the webtext-v3 prereg names these three adds in PROSE and
# numbers none of them. `roster_row` is a required field, so a number had to be
# written: 24/25/26 continue the campaign table in the order §5 lists the adds
# (Qwen2.5-72B-Instruct · Llama-3.1-8B base · Qwen2.5-7B base) — row 24 is
# `V3_SCALE_ADD` above. Nothing keys off the number.
#
# WHAT A ROW BUYS, AND WHAT IT DOES NOT. It buys exactly one thing, the same one
# it buys anywhere in this module: SCAN_GRIDS gains the key, so `MODEL_KEYS =
# SITES ∪ SCAN_GRIDS` accepts it and `collect_mean_states.py --model` will run it.
# A row registers NO site by itself — sites from curves, never fiat, and the
# frozen §5 requires exactly that ("Core models without a registered site at
# freeze receive the site-curve ceremony inside their collection window,
# registered BEFORE any fit of theirs is quoted").
#
# THE CEREMONY HAS NOW RUN FOR BOTH OF THESE KEYS AND LUXIA HAS RULED
# (2026-08-03, on the sibling scan curves; ledger "TWO SITE RULINGS"): both are
# registered in `fit_transport_maps.SITES` and
# `read_composed_predictions.SITE_OF_RECORD`, and each carries a
# `scan_grid_extension` of ONE site — the ruled SAME-SITE pair-read site that the
# computed 12 never visited. The per-row notes below carry the evidence; the
# extension field is what keeps the ratification invariant honest about it (a
# site registered off the effective scan grid is a fiat grid by construction, and
# `fiat_grid_problems` would fail the import). Row 24 above has SINCE had its own
# ceremony too (Luxia, 2026-08-04, sites (58, 63) ⋆ L58) — so all three frozen §5
# CORE adds are now registered — but it needs NO extension, because both of its
# ruled sites are on its computed grid. The extension mechanism is what these two
# rows need and row 24 does not.
#
# ARMS ARE RAW-ONLY, BY DESK RULING (2026-08-03, ledger block of record) — and
# note that this is a RULING and not an inference from the checkpoints: llama-base
# has no chat template and could not run a native arm anyway, but qwen-base SHIPS
# one, and its presence proves nothing (rake M9: a template on a base checkpoint
# is a packaging fact, not an identity). The rule comes from the frozen webtext-v3
# prereg §3.2 instead: "any slot with a base-model endpoint is scored in
# raw::proc_k256 (the parent's arm rule, inherited)". The raw arm is the arm these
# nodes are read in, so it is the arm they are collected in — the pythia-6.9b and
# gpt2-xl precedent, reached here for a different reason.
#
# Architecture facts below are the M17(a) SEPARATION EVIDENCE staged 2026-08-03
# (desk logs `/tmp/claude-output/wtv3-wave1-sibling-arch.log` and
# `…-sibling-discriminators.log`, cited in the wave-1 collection enactor's
# report): read from each checkpoint's own config.json on the node, at the pinned
# revision, in the collection venv. The scan grids are the COMPUTED
# `scan_grid(num_hidden_layers)` values, read from that evidence rather than
# re-derived here — and `scan_grid` reproduces both exactly, which is the point of
# never typing a grid.
V3_BASE_SIBLINGS: tuple[RosterNode, ...] = (
    RosterNode(
        key="llama-3.1-8b-base", model_id="meta-llama/Llama-3.1-8B",
        roster_row=25, arms=("raw",), num_hidden_layers=32, hidden_size=4096,
        weights_dirname="Llama-3.1-8B", checkpoint_identity="base",
        config_sha256="54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd",
        max_position_embeddings=131072,
        # RULED BY LUXIA 2026-08-03 (ledger "TWO SITE RULINGS"): L16 is the §7
        # C4 SAME-SITE pair-read site, matched to instruct `8b`'s L16, and the
        # computed [0.15,0.85] grid (odd sites only on a 32-layer stack) never
        # visits it. Recorded HERE, as the ruled deviation, because
        # `fit_transport_maps.SITES` registers it and the ratification invariant
        # checks registrations against the EFFECTIVE scan grid — see the notes.
        scan_grid_extension=(16,),
        notes="WEBTEXT-V3 CORE ADD (frozen prereg §5): the pretrain PARENT of the "
              "campaign's primary hub `8b` (Llama-3.1-8B-Instruct), which is what "
              "makes the §C4 sibling read a post-training contrast and not a "
              "family comparison. PINNED REVISION "
              "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b — the staging pull of "
              "2026-08-03 fetched that commit and verified every LFS file's "
              "sha256 against the hub at it (a two-sided comparison covering 100% "
              "of weight bytes), with `original/consolidated.00.pth` declared "
              "EXCLUDED (Meta's original-format copy of the same weights; every "
              "metabasis instrument reads the safetensors shards + index). "
              "ARCHITECTURE, read from the checkpoint's own config.json at that "
              "revision: LlamaForCausalLM, 32 decoder layers, hidden_size 4096, "
              "32 attention heads / 8 kv heads (GQA 4:1), "
              "max_position_embeddings 131072, torch_dtype bfloat16. Same depth "
              "and width as the instruct hub, as a pretrain parent must be. "
              "IDENTITY (rakes M9/M17(a), verified independently of the dirname, "
              "and SHOWN to separate rather than assumed to): the confusable "
              "sibling is the INSTRUCT checkpoint already on the store, and all "
              "THREE metadata discriminators separate them — config.json "
              "54acfad3… vs 29e4c210…, generation_config.json e645194d… vs "
              "189fb0c0…, tokenizer_config.json 8004530f… vs 177c7b61… "
              "(special_tokens_map.json separates too: 462d9193… vs 6f38c737…). "
              "The DISCRIMINATOR OF RECORD IS THE CONFIG SHA, the others are "
              "corroborators. CHAT TEMPLATE: ABSENT — this checkpoint has none, "
              "so the native arm does not exist for it at all. "
              "ARMS RAW-ONLY (desk ruling 2026-08-03) — see the group header: the "
              "ruling rests on frozen §3.2's base-endpoint scoring rule, not on "
              "the absent template, which is why it reads identically on the qwen "
              "sibling that HAS one. "
              "SCAN GRID: 32 layers, so the computed 12-site [0.15,0.85] grid is "
              "(5,7,9,11,13,15,17,19,21,23,25,27) — DERIVED by `scan_grid`, never "
              "typed, and identical to the value the staged evidence log "
              "computed. That grid is what the SCAN ran on (12 sites, raw, 36/36 "
              "cells valid). "
              "⚠ ONE RULED EXTENSION, L16, AND IT IS NOT A CURVE SITE: see "
              "`scan_grid_extension` above. L16 was added by Luxia's ruling for "
              "the §7 C4 SAME-SITE sibling read against instruct `8b`'s L16 — "
              "frozen §7 EXCLUDES a sibling pair whose sites differ as "
              "site-confounded, so the pair-read site is a requirement of the "
              "read rather than a finding of the curve. The 32-layer computed "
              "grid is odd-only and cannot contain 16 at all, so no curve could "
              "have produced it. "
              "SITES REGISTERED (13, 15, 16) ⋆ L15 — RULED BY LUXIA 2026-08-03 "
              "on the webtext-v3 sibling scan curve (fits_scan cp2 `488a81d1…`, "
              "36/36 valid, on the FROZEN --splits-artifact membership 920/280): "
              "L15 is the ridge + cka peak (ridge .7224, cka_after .9781), L13 is "
              "kept because it is the PROC-family peak (k128 and k32 both), and "
              "L16 is the pair-read site above. `fit_transport_maps.SITES` and "
              "`read_composed_predictions.SITE_OF_RECORD` carry the same ruling; "
              "the two registries must agree. "
              "⚠ L16 IS NOT BANKED YET: the scan collected the computed 12, so a "
              "default-grid fit resolves a site the bank does not hold until the "
              "ruled ~70s re-collect lands (it joins wave 3 with the ⋆-site "
              "collections). The registry states the RULING; the bank catches up "
              "to it. "
              "WEIGHTS: the metabasis-owned v3 roster tree on the RAID (standing "
              "placement ruling 2026-07-29), a plain directory named by "
              "`weights_dirname`; the full path is passed via --model-path and is "
              "never hardcoded here. A pre-existing `llama-3.1-8b-base` tree on "
              "the same node is byte-identical (same 16 sha256s, already this "
              "revision); the duplicate was staged deliberately so the v3 tree is "
              "self-describing, and collapsing it is a desk/Luxia call."),
    RosterNode(
        key="qwen2.5-7b-base", model_id="Qwen/Qwen2.5-7B",
        roster_row=26, arms=("raw",), num_hidden_layers=28, hidden_size=3584,
        weights_dirname="Qwen2.5-7B", checkpoint_identity="base",
        config_sha256="267ce68584c5f24c3b267d934db2de68dd21d1ca677fb78ed809eb60067f7642",
        max_position_embeddings=131072,
        # RULED BY LUXIA 2026-08-03, exactly as on the llama sibling above: L21
        # is the §7 C4 SAME-SITE pair-read site, matched to instruct `qwen-7b`'s
        # L21, and the computed 28-layer grid does not visit it (it holds 20 and
        # 22, straddling it).
        scan_grid_extension=(21,),
        notes="WEBTEXT-V3 CORE ADD (frozen prereg §5): the pretrain PARENT of the "
              "carried banked node `qwen-7b` (Qwen2.5-7B-Instruct), the second "
              "sibling pair of the §C4 base-vs-instruct read. PINNED REVISION "
              "d149729398750b98c0af14eb82c78cfe92750796, staged and sha-certified "
              "2026-08-03 the same way as its Llama counterpart (every LFS file's "
              "sha256 compared two-sidedly against the hub at that commit; "
              "nothing excluded from this repo). "
              "ARCHITECTURE, read from the checkpoint's own config.json at that "
              "revision: Qwen2ForCausalLM, 28 decoder layers, hidden_size 3584, "
              "28 attention heads / 4 kv heads (GQA 7:1), "
              "max_position_embeddings 131072, torch_dtype bfloat16. "
              "IDENTITY (rakes M9/M17(a)) — THE DISCRIMINATOR OF RECORD IS THE "
              "CONFIG SHA, shown to separate: config.json 267ce685… on this base "
              "checkpoint against 7463bb0e… on the Qwen2.5-7B-Instruct sibling "
              "resident on the shared store. Corroborators: generation_config.json "
              "8c970692… vs 3a8f9087…, tokenizer_config.json c91efca1… vs "
              "5b5d4f65…. ⚠ ONE DISCRIMINATOR IS DEGENERATE HERE AND IS RECORDED "
              "AS SUCH: special_tokens_map.json is ABSENT FROM BOTH trees, so it "
              "separates nothing on this lineage — the mixtral/gemma lesson (M17: "
              "the discriminator is CHOSEN PER LINEAGE and shown, never reused by "
              "reflex) reached for a third time. "
              "⚠ THIS CHECKPOINT SHIPS A CHAT TEMPLATE, AND IT IS NOT EVIDENCE OF "
              "ANYTHING (rake M9): a base checkpoint carrying a template is a "
              "packaging fact. Template PRESENCE is therefore not used to tell "
              "this checkpoint from its instruct sibling, and it does NOT open a "
              "native arm here — arms are raw-only by the 2026-08-03 desk ruling, "
              "which rests on frozen §3.2's base-endpoint scoring rule. This row "
              "is the reason that ruling had to be stated in prereg terms rather "
              "than read off the checkpoints. "
              "SCAN GRID: 28 layers, so the computed 12-site [0.15,0.85] grid is "
              "(4,6,8,10,11,13,15,17,18,20,22,24) — DERIVED by `scan_grid`, never "
              "typed, and identical to the value the staged evidence log "
              "computed. Note the DEDUPE-FREE 12 despite the shallow stack. That "
              "grid is what the SCAN ran on (raw, 36/36 cells valid). "
              "⚠ ONE RULED EXTENSION, L21, AND IT IS NOT A CURVE SITE: see "
              "`scan_grid_extension` above — the §7 C4 SAME-SITE pair-read site "
              "matched to instruct `qwen-7b`'s L21, required by the read (a pair "
              "whose sites differ is quoted site-confounded and EXCLUDED) rather "
              "than found by the curve, which straddles it at 20 and 22. "
              "SITES REGISTERED (18, 20, 22, 21) ⋆ L20 — RULED BY LUXIA "
              "2026-08-03 on the webtext-v3 sibling scan curve (fits_scan cp2 "
              "`cc9d474c…`, 36/36 valid, FROZEN --splits-artifact membership): "
              "L20 is the UNANIMOUS peak — every fit family agrees, .7796 — and "
              "18/22 are its curve bracket; L21 is the pair-read site. The tuple "
              "is written in the ORDER LUXIA RULED IT (bracket first, pair-read "
              "site appended), not re-sorted, so the registry reads as the ruling "
              "reads; nothing downstream depends on the order. CURVE CAVEAT "
              "RECORDED WITH THE PICK: the proc family COLLAPSES over L4–L8 "
              "(shallow-half divergence); all families agree from L15 on, which "
              "is the region the ruling is drawn from. "
              "⚠ L21 IS NOT BANKED YET — same state as the llama sibling's L16: "
              "the ruled re-collect joins wave 3, and until it lands a "
              "default-grid fit resolves one site the bank does not hold. "
              "WEIGHTS: the metabasis-owned v3 roster tree on the RAID, flat "
              "directory layout, 4 shards, index-complete; the full path is "
              "passed via --model-path and is never hardcoded here."),
)

ROSTER: dict[str, RosterNode] = {
    n.key: n for n in WAVE1 + HUB_RUNGS_2 + MOE_CHAT + BIG_CHAIN_SINGLE_CARD
    + BIG_CHAIN_MULTICARD + CARRIED_BANKED + BEHAVIORAL_TIER + V3_SCALE_ADD
    + V3_BASE_SIBLINGS}

#: model key -> the grid it was actually collected and curve-scanned on. This is
#: what `--sites` should carry for a scan collection, and what `--tgt-sites`
#: should carry for the scan fit. Normally the computed 12 sites; for a node with
#: a ruled extension (pythia-6.9b, gpt2-xl, gemma3-27b) it is the EFFECTIVE grid,
#: because that is what the bank holds and what reproduces the published curve.
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
