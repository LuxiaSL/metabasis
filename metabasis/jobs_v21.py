"""ONE parameterized jobs-v21 template family, and the harness that proves it.

WHY THIS MODULE EXISTS. Every collection wave hand-writes a header variant of
the same node-side job script: the four FD-gated vector builds under
`jobs-v21/` share ~120 lines of paths, gates, markers, build invocation, sha
manifest and exit discipline, and differ in their per-model EVIDENCE — which
checkpoint, which sibling it is confusable with, which site the registries
ratified, which memory shape it loads under. The ledger's signal was "a
header-only variant ×3"; the rung arrived (a fourth), and the rule is that the
skeleton lives here and the evidence stays data.

────────────────────────────────────────────────────────────────────────────────
THE DIVISION OF LABOUR, AND WHY IT IS DRAWN HERE
────────────────────────────────────────────────────────────────────────────────
**The TEMPLATE owns semantics.** Anything that decides whether a job runs, what
it runs, or what it writes — the preflight gates, the `python -m` invocation and
its flags, the `PIPESTATUS` discipline, the success/refusal markers, the
refuse-to-overwrite and wrong-tree guards. One copy, one place to fix.

**The SPEC owns evidence.** Anything a reader has to be able to AUDIT against a
checkpoint — config digests, sibling discriminators, the site of record, the
prose that explains why THIS build is admissible. A template must never
manufacture that text: a generated identity paragraph is a paragraph nobody
verified. So the header, the explanatory `print(...)` lines and the per-model
constants are carried as data, and the rendered script is byte-for-byte the
script the wave would have hand-written.

That division is what makes the byte-comparison harness meaningful. `compare()`
renders a spec and diffs it against the DEPLOYED script of record; equality is
the claim, and a difference is a defect to adjudicate rather than a diff to
eyeball. Rendering is CPU-only — no torch, no GPU, no node — so the proof runs
at the desk before anything is deployed.

────────────────────────────────────────────────────────────────────────────────
THE CORPUS IS A PARAMETER, DELIBERATELY
────────────────────────────────────────────────────────────────────────────────
Nothing here hardcodes corpus v2.1. The manifest RELPATH, its SHA, the shell
variable that holds it, the label that names it in a log line and the `corpus=`
tag in the success marker are five separate spec fields, because the deployed
v2 and v2.1 ancestors differ in all five and the v3 basis ruling is open. The
selftest proves this the only way that proves anything: it renders the v2
ancestor and the v2.1 script of record FROM THE SAME TEMPLATE and requires both
to come out byte-exact.

────────────────────────────────────────────────────────────────────────────────
THE ×1.6 HEURISTIC, FOLDED IN
────────────────────────────────────────────────────────────────────────────────
Four deployed vector scripts carry `weights × 1.6 / n_cards` inline in a python
heredoc — the fifth and sixth copies of the arithmetic `metabasis.capacity`
exists to be. `memory_check` selects between them:

    "inline-flat"      the deployed block, reproduced verbatim. FIDELITY mode:
                       what the byte-comparison harness renders, because the
                       claim is about the script that actually ran.
    "capacity-module"  the same decision taken by `metabasis.capacity`, which
                       computes BOTH bounds, takes the max and names which one
                       bound. The DEFAULT for anything newly rendered.

The two modes differ in exactly one block, and `metabasis.jobs_v21
--conversion-diff` prints it, so the conversion is reviewable as a diff rather
than as a claim.

────────────────────────────────────────────────────────────────────────────────
THE THREAD COUNT, AND WHY IT IS A SECOND FIDELITY AXIS
────────────────────────────────────────────────────────────────────────────────
By Luxia's ruling of 2026-08-01 `OMP_NUM_THREADS=8` is the standing default
everywhere, because eigh is bitwise-deterministic at a FIXED thread count and
its bytes differ ACROSS counts — so the number a job exports is part of what
its vectors ARE, and every builder now stamps the count it actually ran at.

Every DEPLOYED script of record exports `OMP_NUM_THREADS=1`, and the 7/7
byte-identity proof against those scripts is of record. So the ruled default
enters as a PARAMETER on the line that already existed (`ThreadPolicy.ruled`
vs `ThreadPolicy.deployed`), plus a named comment that only a NEW-STYLE render
emits. `render(..., fidelity=True)` — which is what `compare()` uses — emits
the ancestor's own number and not one new line, so the proof is untouched.
`fidelity` is a separate switch from `memory_check` on purpose: they answer
different questions, and folding one into the other would mean every future
ruling that touches a shared line had to be smuggled through the memory mode.

────────────────────────────────────────────────────────────────────────────────
MIRRORED ANCESTRY, AND WHAT IS NOT MIRRORED
────────────────────────────────────────────────────────────────────────────────
A lane is TRUSTED when a deployed script of record exists to diff against.
`vector-generic`, `vector-model` and `vector-wrapper` are mirrored under
`staging/node-jobs-mirror/` and are proved byte-exact. `collect` and `fits` are
NOT: their skeletons are derived from the CLIs of record
(`collect_mean_states`, `fit_transport_maps`) and the documented job
conventions, and they are refused by `render()` unless the caller passes
`allow_unmirrored=True` — a rendered script carries a banner saying so. An
unproved lane that renders silently is worse than no lane at all.

NO SUBMISSION HAPPENS HERE. The `submit` lane emits the `heimdall submit` line
in the CLI form the deployed scripts use, to a file, for a human to read and
fire. This module never shells out, never touches a node, and never submits.

WHERE THE FIDELITY FIXTURES LIVE
────────────────────────────────────────────────────────────────────────────────
The specs that reproduce the deployed scripts embed their VERBATIM prose, and
that prose names a cluster — not only in paths (which tokenize) but in
vocabulary, which does not (rake M46: filenames AND content are the
sanitization surface). By desk ruling 2026-08-01 the whole fidelity set lives
OUTSIDE this repo, in the gitignored `staging/jobs-v21-specs/`, and only the
synthetic examples are tracked. Everything that needs it takes `--spec-dir`,
and every check that cannot run without it says so BY NAME rather than passing
quietly.

Run (repo root, PYTHONPATH=.):
  python -m metabasis.jobs_v21 --selftest
  python -m metabasis.jobs_v21 --selftest --spec-dir <STAGING>/jobs-v21-specs \\
      --mirror-dir <MIRROR_ROOT> --deployment <STAGING>/jobs-v21-deployment.json
  python -m metabasis.jobs_v21 --list [--spec-dir DIR]
  python -m metabasis.jobs_v21 --render <spec> [--out-dir DIR]
                               [--memory-check capacity-module] [--fidelity]
  python -m metabasis.jobs_v21 --conversion-diff <spec>
  python -m metabasis.jobs_v21 --compare-mirror <MIRROR_ROOT> \\
      --spec-dir <STAGING>/jobs-v21-specs [--json]
  python -m metabasis.jobs_v21 --fire-line <spec> --node <NODE>
                               --jobs-dir <NODE_JOBS_DIR> [--after JOB_ID]
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field, ValidationError, model_validator

from metabasis.threads import RULED_OMP_NUM_THREADS

#: Templates and the spec fixtures for the deployed scripts of record.
TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates" / "jobs_v21"
SPEC_ROOT = TEMPLATE_ROOT / "specs"

#: Lanes whose rendered output has a deployed script of record to diff against.
#: Membership here is the whole difference between "proved" and "plausible".
MIRRORED_LANES: frozenset[str] = frozenset(
    {"vector-generic", "vector-model", "vector-wrapper"})

#: The banner a lane with no mirrored ancestor carries into its own output, so
#: the warning travels with the artifact and not only with the conversation
#: that produced it.
UNMIRRORED_BANNER: tuple[str, ...] = (
    "# ⚠ NO MIRRORED ANCESTOR — THIS SCRIPT HAS NOT BEEN BYTE-COMPARED.",
    "#   Its skeleton is derived from the CLI of record plus the documented job",
    "#   conventions, NOT from a deployed script that has run. Diff it against",
    "#   the deployed job of record before it is fired, and pull that script",
    "#   back into staging/node-jobs-mirror/ so the next render is proved.",
)


#: `<<TOKEN>>` in a spec string — a DEPLOYMENT FACT held out of the repo.
#: Distinct from the template's `{{ }}` and from shell's `${ }` on purpose: the
#: three substitutions happen at different times, to different audiences, and a
#: shared syntax would let one silently do another's job.
PLACEHOLDER = re.compile(r"<<([A-Z][A-Z0-9_]*)>>")

#: Where the deployment overlay lives by default: beside the pulled-back node
#: mirror, in the gitignored staging tree. Repo policy is that node names,
#: usernames and store paths never push, and a spec that names a weights store
#: is a spec that names one — so the specs carry TOKENS and the values stay here.
DEPLOYMENT_DEFAULT = (Path(__file__).resolve().parents[1] / "staging"
                      / "jobs-v21-deployment.json")

#: Neutral values the selftest renders under. They are NOT the cluster's: the
#: selftest proves the machinery and every refusal, and BYTE-IDENTITY against
#: the deployed scripts is a separate, named check that requires the real
#: overlay and the real mirror (both local-only).
SELFTEST_DEPLOYMENT: dict[str, str] = {
    "PROJECT_ROOT": "$HOME/<project>",
    "VENV_ACTIVATE": "$HOME/<project>/.venv/bin/activate",
    "SHARED_WEIGHTS": "<shared-weights>",
    "RAID": "<raid>",
    "NODE_A": "<node-a>",
}


class TemplateError(RuntimeError):
    """A template could not be rendered as written.

    Raised only for a MALFORMED template or an incomplete context — an unclosed
    block, a placeholder no context key answers, a splice whose value is not a
    list of lines. Never for "the rendered script would not work": this module
    cannot know that, and pretending otherwise would put a guess where a proof
    belongs.
    """


class SpecError(RuntimeError):
    """A spec cannot be turned into a job as asked."""


class Deployment(BaseModel):
    """The cluster-shaped values a spec holds out of the repo, as `<<TOKEN>>`.

    A spec in this repo says `<<SHARED_WEIGHTS>>/Llama-3.3-70B-Instruct`, and
    the store's real path lives in a gitignored overlay beside the node mirror.
    That is not obfuscation — it is the same reason the mirror itself is
    gitignored: the campaign's code is meant to travel and its cluster is not.

    A MISSING token is a REFUSAL, never an empty string. A job script that
    silently rendered `/Llama-3.3-70B-Instruct` would fail on the node, in a
    queue slot, with a path error that reads like a staging mistake.
    """
    tokens: dict[str, str]

    @model_validator(mode="after")
    def _tokens_are_usable(self) -> "Deployment":
        for name, value in self.tokens.items():
            if not PLACEHOLDER.fullmatch(f"<<{name}>>"):
                raise ValueError(
                    f"{name!r} is not a token name (UPPER_SNAKE, leading "
                    f"letter) — a token nothing can reference resolves nothing")
            if not value:
                raise ValueError(
                    f"{name}: empty. An empty deployment value renders a path "
                    f"with a hole in it and fails on the node rather than here")
            if PLACEHOLDER.search(value):
                raise ValueError(
                    f"{name}={value!r} contains a placeholder itself; "
                    f"substitution is ONE pass, deliberately, so a self-"
                    f"referential overlay cannot loop or resolve order-"
                    f"dependently")
        return self

    def resolve(self, value: str) -> str:
        return PLACEHOLDER.sub(
            lambda m: self.tokens.get(m.group(1), m.group(0)), value)


def load_deployment(path: Path) -> Deployment:
    """Read a deployment overlay. Its absence is a NAMED refusal, not a crash."""
    try:
        raw = json.loads(Path(path).read_bytes())
    except OSError as exc:
        raise SpecError(
            f"no deployment overlay at {path} ({exc}). The specs carry "
            f"<<TOKEN>> placeholders because node names, usernames and store "
            f"paths do not travel with this repo; the values live in a "
            f"gitignored overlay beside the node mirror. Write one, or pass "
            f"--deployment") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path}: is not valid JSON ({exc})") from exc
    try:
        return Deployment(tokens=raw.get("tokens", raw))
    except ValidationError as exc:
        raise SpecError(f"{path}: {exc}") from exc


# ---------------------------------------------------------------- the renderer
#: `{{ name }}` — inline scalar. Deliberately NOT `${...}`: shell scripts are
#: full of `${VAR}` and a template language that collides with its own output
#: is a template language that silently rewrites shell.
_SCALAR = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
#: `{{*name}}` alone on a line — splice a list of lines VERBATIM (no re-indent:
#: the spliced lines are evidence text and their indentation is part of it).
_SPLICE = re.compile(r"^\s*\{\{\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}\s*$")
_IF = re.compile(r"^\s*\{\{#if\s+([A-Za-z_][A-Za-z0-9_]*)\s*\}\}\s*$")
_ELSE = re.compile(r"^\s*\{\{#else\s*\}\}\s*$")
_ENDIF = re.compile(r"^\s*\{\{/if\s*\}\}\s*$")

RenderContext = dict[str, Any]


def _substitute(line: str, ctx: RenderContext, where: str) -> str:
    """Fill every `{{ name }}` on one line, or say which name had no answer."""
    def one(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in ctx:
            raise TemplateError(
                f"{where}: no context value for {{{{{key}}}}}. A template that "
                f"renders an unanswered placeholder as empty produces a job "
                f"script with a silently missing path or digest — the one "
                f"failure mode a byte-comparison would not catch, because both "
                f"sides would be wrong the same way")
        value = ctx[key]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TemplateError(
                f"{where}: {{{{{key}}}}} resolves to {type(value).__name__}; "
                f"only str/int substitute inline (a list splices with "
                f"{{{{*{key}}}}}, a bool switches with {{{{#if {key}}}}})")
        return str(value)
    return _SCALAR.sub(one, line)


def _render_lines(lines: Sequence[str], ctx: RenderContext, start: int,
                  emit: bool, where: str) -> tuple[list[str], int]:
    """Render from `start` until end-of-input or the matching `{{/if}}`.

    `emit=False` walks a not-taken branch WITHOUT substituting into it, so an
    unused branch cannot fail a render on a context key only the other branch
    needs. It is still PARSED, so a malformed block inside a dead branch is
    still an error — a template nobody can render both ways is a template whose
    second way has never been checked.
    """
    out: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if _ENDIF.match(line):
            return out, i
        m_if = _IF.match(line)
        if m_if:
            key = m_if.group(1)
            if key not in ctx:
                raise TemplateError(
                    f"{where}:{i + 1}: {{{{#if {key}}}}} has no context value; a "
                    f"switch nobody set defaults to nothing, and 'nothing' is a "
                    f"branch that was never chosen")
            taken = bool(ctx[key])
            then_lines, j = _render_lines(lines, ctx, i + 1, emit and taken, where)
            else_lines: list[str] = []
            if j < len(lines) and _ELSE.match(lines[j]):
                else_lines, j = _render_lines(lines, ctx, j + 1,
                                              emit and not taken, where)
            if j >= len(lines) or not _ENDIF.match(lines[j]):
                raise TemplateError(
                    f"{where}:{i + 1}: {{{{#if {key}}}}} is never closed by "
                    f"{{{{/if}}}}")
            if emit:
                out.extend(then_lines if taken else else_lines)
            i = j + 1
            continue
        if _ELSE.match(line):
            return out, i
        m_splice = _SPLICE.match(line)
        if m_splice:
            key = m_splice.group(1)
            if key not in ctx:
                raise TemplateError(
                    f"{where}:{i + 1}: no context value for {{{{*{key}}}}}")
            value = ctx[key]
            if not isinstance(value, list) or any(not isinstance(x, str)
                                                  for x in value):
                raise TemplateError(
                    f"{where}:{i + 1}: {{{{*{key}}}}} must splice a list of "
                    f"str lines, got {type(value).__name__}")
            if emit:
                out.extend(value)
            i += 1
            continue
        if emit:
            out.append(_substitute(line, ctx, f"{where}:{i + 1}"))
        i += 1
    return out, i


def render_template(text: str, ctx: RenderContext, where: str = "<template>") -> str:
    """Render one template. Deterministic, pure, and newline-exact.

    The output ends with exactly one trailing newline whenever the template did,
    because a shell script that loses its final newline is a different file to
    `sha256sum` and an identical one to every human reading it — precisely the
    class of difference a byte-comparison exists to refuse.
    """
    lines = text.split("\n")
    trailing = ""
    if lines and lines[-1] == "":
        lines = lines[:-1]
        trailing = "\n"
    rendered, consumed = _render_lines(lines, ctx, 0, True, where)
    if consumed < len(lines):
        raise TemplateError(
            f"{where}:{consumed + 1}: stray {lines[consumed].strip()!r} — a "
            f"{{{{/if}}}} or {{{{#else}}}} with no {{{{#if}}}} above it")
    return "\n".join(rendered) + trailing


# ---------------------------------------------------------------- the spec
class CorpusBasis(BaseModel):
    """WHICH corpus a job asserts, as five separable facts.

    They are separate because the deployed v2 and v2.1 ancestors differ in all
    five, and the v3 basis ruling is OPEN. A single `corpus="v2.1"` field would
    have to grow a decoder ring the moment v3 lands; five plain fields never do.
    """
    manifest_relpath: str = Field(
        default="corpus/corpus_manifest.json",
        description="path under the arm root, named EXACTLY as on disk — the "
                    "checksum-list file is a different artifact and the "
                    "ambiguity has caused a near-HALT before")
    sha256: str = Field(description="byte-sha of the manifest the job asserts")
    shell_var: str = Field(
        default="CORPUS_MANIFEST_SHA",
        description="the shell variable that carries the digest. A PARAMETER "
                    "because the deployed scripts bake a basis version into "
                    "the NAME (CORPUS_V21_SHA / CORPUS_V2_SHA) and byte-"
                    "fidelity against them requires reproducing it")
    label: str = Field(description="how the basis is named in log prose, e.g. "
                                   "'corpus-v2.1'")
    tag: str = Field(description="the `corpus=` value in the success marker, "
                                 "e.g. 'v21'")
    n_texts: Optional[int] = Field(
        default=None, description="entry count the preflight asserts the loaded "
                                  "corpus has; None => the count is not asserted")

    @model_validator(mode="after")
    def _digest_is_a_digest(self) -> "CorpusBasis":
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError(
                f"corpus sha256 {self.sha256!r} is not 64 lowercase hex. A "
                f"TRUNCATED digest in a preflight is worse than none: it "
                f"passes on more checkpoints than the operator thinks (rake "
                f"M40 — full digests only, never a padded prefix)")
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", self.shell_var):
            raise ValueError(f"{self.shell_var!r} is not a shell variable name")
        return self


class NodePaths(BaseModel):
    """Where the job stands on the node.

    NO DEFAULTS, deliberately. This module ships no node-side path at all: the
    project tree, the venv and the bank root are DEPLOYMENT facts and they live
    in the spec, which is the artifact the sanitization sweep looks at. A
    default here would put a cluster's directory layout into shared code, where
    nobody would think to look for it and where it would go stale silently.
    """
    root: str = Field(
        description="the project tree on the node. Keep it a shell expansion "
                    "($HOME/…): the rendered script must not carry a resolved "
                    "username, and $HOME is also the only form that survives "
                    "being run as a different cluster user")
    code_dirname: str = Field(description="the deployed code tree under `root`")
    arm_dirname: str = Field(description="the arm/bank tree under `root`")
    venv_activate: str = Field(description="the venv activate script to source")
    bankroot: str = Field(
        default="",
        description="the RAID tree this wave writes under (standing placement "
                    "rule 2026-07-29). Empty => logs live under the arm tree, "
                    "which is what the generic lane does")

    @model_validator(mode="after")
    def _no_resolved_home(self) -> "NodePaths":
        for name in ("root", "venv_activate"):
            value = getattr(self, name)
            if value.startswith("/home/") or value.startswith("/Users/"):
                raise ValueError(
                    f"{name}={value!r} carries a resolved home directory. Job "
                    f"scripts travel into the repo and into reports; the "
                    f"sanitization rule wants $HOME, and $HOME is also the only "
                    f"form that survives being run as a different cluster user")
        return self


MemoryCheck = Literal["inline-flat", "capacity-module", "none"]


class MemoryPolicy(BaseModel):
    """How the preflight decides the model fits, and over how many cards.

    `inline-flat` reproduces the arithmetic the deployed scripts carry inline;
    `capacity-module` delegates to `metabasis.capacity`, which computes the flat
    bound AND the M19-derived structural bound and takes the max. The two agree
    on every deployed case (the flat bound was the stricter one), which is why
    the conversion is a diff and not a re-derivation.
    """
    mode: MemoryCheck = Field(default="capacity-module")
    shard_across: str = Field(
        default="",
        description="the builder's COMMA DEVICE LIST (never the count form the "
                    "collector accepts). Empty => single card")
    multiplier: float = Field(default=1.6, gt=0.0)

    @property
    def n_cards(self) -> int:
        return len([d for d in self.shard_across.split(",") if d.strip()]) or 1

    @property
    def is_sharded(self) -> bool:
        return bool(self.shard_across.strip())


class ThreadPolicy(BaseModel):
    """How many BLAS/OpenMP threads a rendered job exports — and which number.

    THE RULING (Luxia, 2026-08-01): `OMP_NUM_THREADS=8` is the standing default
    everywhere, and the effective count is recorded in every build stamp as
    part of instrument identity. eigh is bitwise-deterministic at a FIXED
    thread count and its bytes differ ACROSS counts, so the exported number is
    not a performance knob — it is part of what an artifact IS.

    TWO NUMBERS, DELIBERATELY. Every deployed script of record exports
    `OMP_NUM_THREADS=1`, and the 7/7 byte-identity proof against those scripts
    is of record. If the ruled default simply replaced the literal, that proof
    would break for a reason that is not a defect. So `deployed` is what
    FIDELITY renders (the ancestor's own number, plus nothing) and `ruled` is
    what a NEW-STYLE render exports — one template, both scripts, the same
    discipline the corpus basis already travels under.
    """
    ruled: int = Field(
        default=RULED_OMP_NUM_THREADS, ge=1,
        description="what a NEW-STYLE render exports. The ruled default of "
                    "record; a spec overrides it only by ruling, and the "
                    "rebuilt vectors' bytes move when it does")
    deployed: int = Field(
        default=1, ge=1,
        description="what the DEPLOYED ancestor exports, reproduced verbatim "
                    "by FIDELITY mode. Defaults to 1 because every deployed "
                    "script of record carries `export OMP_NUM_THREADS=1`, so "
                    "the fidelity fixtures need no new field to keep passing")
    ruling_note: bool = Field(
        default=True,
        description="emit the named comment above the export in NEW-STYLE "
                    "renders, so a reader of a job script learns WHY the "
                    "number is load-bearing without leaving the script. Never "
                    "emitted in fidelity mode: it is new text, and new text is "
                    "exactly what a byte-comparison exists to refuse")


#: The ruling comment a NEW-STYLE render carries above its export. Held here
#: rather than in the templates so the four lanes cannot drift apart in the one
#: block whose whole job is to say the same thing everywhere.
THREAD_RULING_NOTE: tuple[str, ...] = (
    "# OMP_NUM_THREADS is the standing default of record (Luxia ruling",
    "# 2026-08-01), not a performance knob: eigh is bitwise-deterministic at a",
    "# FIXED thread count and its bytes differ ACROSS counts, so the count is",
    "# part of the instrument identity every builder now stamps. Changing it",
    "# changes the vectors' bytes; the deployed pre-ruling scripts exported 1,",
    "# and any comparison across the boundary quotes both counts.",
)


class _LaneBase(BaseModel):
    """Fields every lane carries. `name` is the rendered filename."""
    name: str = Field(description="the rendered script's filename")
    threads: ThreadPolicy = Field(
        default_factory=ThreadPolicy,
        description="the exported thread count, in both modes. Every lane "
                    "carries it (vector-wrapper execs the generic instrument "
                    "and exports nothing itself, so its template reads none of "
                    "it) — one field, so a new lane cannot forget the ruling")
    header: list[str] = Field(
        default=[],
        description="the header comment block, VERBATIM and including its "
                    "leading '#'. Carried as data because it states per-model "
                    "evidence (which sibling, which ruling, which digest) that "
                    "a template must never manufacture")

    @model_validator(mode="after")
    def _header_is_comment(self) -> "_LaneBase":
        bad = [ln for ln in self.header if ln and not ln.startswith("#")]
        if bad:
            raise ValueError(
                f"{self.name}: header line(s) {bad[:2]} are not comments. The "
                f"header splices in directly above `set -u`; a non-comment line "
                f"there is executable code entering a job through the one field "
                f"nobody reviews as code")
        return self


class VectorGenericSpec(_LaneBase):
    """The env-driven instrument of record: `jobs-v21/build-vec-v21.sh`.

    Takes MB_MODEL/MB_MODEL_PATH/MB_SITE/MB_ARM from the environment and knows
    nothing about any particular checkpoint — which is why one template renders
    both it and its corpus-v2 ancestor, and why that pair is the v3-readiness
    proof.
    """
    lane: Literal["vector-generic"] = "vector-generic"
    paths: NodePaths
    corpus: CorpusBasis
    log_stem: str = Field(description="log filename stem, e.g. 'build-vec-v211' "
                                      "— reproduced as deployed, typo included")
    marker: str = Field(description="success/refusal marker stem, e.g. "
                                    "'BUILD-VEC-V21'")


class VectorWrapperSpec(_LaneBase):
    """A checkpoint-identity gate that execs the unchanged generic instrument.

    The mixtral shape. It exists because a checkpoint on the SHARED store can be
    mutated under the job (rake M9), so identity is asserted immediately before
    the load rather than assumed — and because the build itself must stay
    byte-for-byte the path its siblings took, the wrapper adds a gate and
    changes nothing else.
    """
    lane: Literal["vector-wrapper"] = "vector-wrapper"
    paths: NodePaths
    model: str
    model_path: str
    site: str
    arm: str = "native"
    jobs_dirname: str = Field(default="jobs-v21")
    target_script: str = Field(default="build-vec-v21.sh")
    constants: list[str] = Field(
        default=[], description="the identity constants block, verbatim")
    checks: list[str] = Field(
        default=[], description="the identity assertions, verbatim shell. This "
                                "lane's whole content IS its evidence, so it is "
                                "data end to end; the template contributes the "
                                "hand-off, which is the part that must not vary")


class VerdictStyle(BaseModel):
    """How the FD-gate verdict is read back OUT of the artifact.

    ⚠ Exit 0 is not the verdict. The verdict is `PASSES_FD_GATE` inside the gate
    json, and every deployed script prints it explicitly whether it passed or
    not — which is why this block exists at all rather than trusting `rc`.
    """
    fields: Literal["explicit", "key-loop"] = "key-loop"
    rung_caption: str = Field(
        default="the FD U-curve; the BEST rung is the meaningful number")
    note: list[str] = Field(
        default=[],
        description="the closing NOTE — the dense band or the MoE framing. "
                    "SURFACED for the desk, never adjudicated in the job, so "
                    "the text is evidence and lives here")


class VectorModelSpec(_LaneBase):
    """One FD-gated per-model vector build: the shape the waves hand-write.

    `jobs-v21/build-vec-v21-<model>.sh` — header prose, constants, a python
    preflight, the build, a verdict read back from the artifact, a sha manifest
    and an exit that distinguishes "no vector because the gate did not pass"
    from "no vector because something broke".

    WHY `preflight_body` IS DATA AND NOT TEMPLATE. Rake M17 is explicit that the
    checkpoint discriminator is CHOSEN PER LINEAGE and SHOWN to separate the
    confusable pair before it is trusted — mixtral's config.json is byte-
    identical between base and instruct, gemma's config IS the discriminator,
    DSV3 degenerates on every metadata field and only the weights separate. A
    template that manufactured an identity block would manufacture exactly the
    reflex M17 exists to forbid, and the generated check would be one nobody
    verified against a checkpoint. So the per-lineage assertions stay verbatim
    evidence, and this model asserts their LOAD-BEARING PROPERTIES mechanically
    instead: the site must come from the registries, the positive config digest
    must be compared, the derived site must be written for the shell to read,
    something must be able to refuse, and the memory decision must NOT be in
    there (it belongs to the template, which is where the estimator conversion
    lands).
    """
    lane: Literal["vector-model"] = "vector-model"
    paths: NodePaths
    corpus: CorpusBasis
    model: str
    model_path: str
    arm: str = "native"
    expect_site: str = Field(description="the ruled site, asserted in-preflight "
                                         "against the registry-derived one")
    expect_config_sha: str
    marker: str = Field(description="marker stem, e.g. 'BUILD-VEC-V21-3370B'")
    log_stem: str = Field(default="build-vec-v21")
    sitefile_stem: str = Field(default="derived-vec-site")
    constants: list[str] = Field(
        default=[], description="constants between EXPECT_CONFIG_SHA and the "
                                "corpus digest, verbatim")
    banner: list[str] = Field(
        default=[], description="the pre-preflight `echo` banner, verbatim")
    compact_cvd_gate: bool = Field(
        default=False, description="the two-line CUDA_VISIBLE_DEVICES refusal "
                                   "rather than the three-line one")
    blank_before_corpus_gate: bool = Field(default=False)
    preflight_argv: list[str] = Field(
        description="the shell lines that pass argv into the preflight heredoc")
    preflight_unpack: list[str] = Field(
        description="the python tuple-unpack of that argv")
    preflight_body: list[str] = Field(
        description="the per-lineage preflight, verbatim: site derivation, "
                    "M9/M17 identity, corpus assertion, G48, tree guards. "
                    "Evidence, not boilerplate — see the class docstring")
    memory: MemoryPolicy = MemoryPolicy()
    memory_section_comment: list[str] = Field(
        default=[], description="the memory block's own comment header, verbatim")
    memory_cards_gate: list[str] = Field(
        default=[],
        description="the single-card refusal body (inline mode only); the "
                    "sharded refusal is uniform and lives in the template")
    memory_needs_index: bool = Field(
        default=False,
        description="True when the weights size is NOT already in scope from an "
                    "earlier index read and the memory block must read it")
    memory_pass_label: str = Field(
        default="the input-gradient pass",
        description="how the multiplier's purpose is named in the log line")
    memory_need_comment: str = Field(
        default="", description="trailing comment on the `need = …` line")
    build_echo: str = Field(description="the `=== BUILD … ===` line, verbatim")
    verdict: VerdictStyle = VerdictStyle()
    verdict_lead: list[str] = Field(
        default=[], description="comment lines above the verdict heredoc")
    ok_suffix: str = Field(
        default="out=$OUT",
        description="the tail of the JOB-OK marker after `corpus=<tag>`")

    @model_validator(mode="after")
    def _load_bearing(self) -> "VectorModelSpec":
        if not re.fullmatch(r"\d+", self.expect_site):
            raise ValueError(
                f"{self.model}: expect_site {self.expect_site!r} is not a "
                f"decoder-layer index. Site L is a forward_pre_hook on "
                f"decoder_layers[L]; a non-integer there is a ruling nobody "
                f"transcribed")
        if not re.fullmatch(r"[0-9a-f]{64}", self.expect_config_sha):
            raise ValueError(
                f"{self.model}: expect_config_sha is not 64 lowercase hex. The "
                f"POSITIVE config assertion is the guard that excludes every "
                f"other checkpoint on its own; a truncated one does not (M40)")
        if self.memory.is_sharded and self.memory.mode == "none":
            raise ValueError(
                f"{self.model}: a sharded build with no memory check. The fan-"
                f"out is the whole reason the check exists — a whole-layer "
                f"pipeline map cannot go below its worst card")
        body = "\n".join(self.preflight_body)
        for needle, why in (
                ("SITE_OF_RECORD",
                 "the site must be DERIVED from the registry, never typed: "
                 "EXPECT_SITE is the assertion, not the source"),
                ("expect_sha",
                 "the POSITIVE config-digest comparison is the guard that "
                 "excludes every other checkpoint on its own"),
                ("sitefile",
                 "the derived site must be written out for the shell to read, "
                 "or the build runs at a site the preflight never approved"),
                ("sys.exit(2)",
                 "a preflight that cannot refuse is a preamble")):
            if needle not in body:
                raise ValueError(
                    f"{self.model}: preflight_body never mentions {needle!r} — "
                    f"{why}")
        for needle, why in (
                ("mem_get_info",
                 "the memory decision belongs to the template, which is where "
                 "the metabasis.capacity conversion lands; an inline copy here "
                 "would be the sixth one and would not be converted with the "
                 "others"),
                ("device_count", "same: the fan-out check is the template's"),
                ("--allow-fd-gate-not-passed",
                 "the FD gate is NEVER waived — a build whose gate does not "
                 "pass must write no vector rather than bank one that does not "
                 "exist")):
            if needle in body:
                raise ValueError(f"{self.model}: preflight_body carries "
                                 f"{needle!r} — {why}")
        return self


class CollectSpec(_LaneBase):
    """Collect + fresh-process bitwise spot-replay in ONE job — the standing gate.

    ⚠ UNMIRRORED. The deployed `scan-*.sh` family has not been pulled back into
    `staging/node-jobs-mirror/`, so this lane has no script of record to be
    diffed against and `render()` refuses it without `allow_unmirrored=True`.
    Its flags come from `collect_mean_states`'s own argparse, which is the CLI
    of record; its shape comes from the documented convention. That is enough to
    review and NOT enough to fire.
    """
    lane: Literal["collect"] = "collect"
    paths: NodePaths
    corpus: CorpusBasis
    model: str
    model_path: str
    arms: str = "native,raw"
    sites: str = Field(description="comma grid — DERIVED from the registry by "
                                   "the caller, never typed by hand")
    spot_replay_k: int = Field(default=3, ge=1)
    max_seq_len: Optional[int] = None
    shard_across: str = ""
    assert_multi_device: bool = False
    marker: str = Field(default="SCAN-V21")
    log_stem: str = Field(default="scan-v21")

    @model_validator(mode="after")
    def _gate_is_real(self) -> "CollectSpec":
        if self.spot_replay_k < 3:
            raise ValueError(
                f"{self.model}: --spot-replay {self.spot_replay_k} is below the "
                f"standing K>=3 gate. The gate is what certifies the forward is "
                f"deterministic for this bank; a weaker one certifies less "
                f"while looking like it certifies the same")
        if self.assert_multi_device and not self.shard_across.strip():
            raise ValueError(
                f"{self.model}: --assert-multi-device with no --shard-across "
                f"asserts a fan-out nobody asked for")
        return self


class FitSpec(_LaneBase):
    """A pair fit over banked states. 0 GPU, CPU spine.

    ⚠ UNMIRRORED, for the same reason as `CollectSpec`: `fit-pair-v21.sh` is
    deployed but not mirrored. The env-variable convention below
    (MB_SRC/MB_SRC_SITE/MB_TGT/MB_TGT_SITE/MB_ARMS) is the one the batch-4 chain
    recorded; the flags are `fit_transport_maps`'s own.
    """
    lane: Literal["fits"] = "fits"
    paths: NodePaths
    source_model: str
    target_model: str
    src_sites: str = ""
    tgt_sites: str = ""
    arms: str = "native,raw"
    n_null: Optional[int] = None
    k_grid: str = ""
    fits_dirname: str
    marker: str = Field(default="FIT-PAIR-V21")
    log_stem: str = Field(default="fit-pair-v21")


JobSpec = Annotated[
    Union[VectorGenericSpec, VectorModelSpec, VectorWrapperSpec, CollectSpec,
          FitSpec],
    Field(discriminator="lane")]


class _SpecEnvelope(BaseModel):
    """What a spec file on disk holds. One spec, discriminated by `lane`."""
    spec: JobSpec


# ---------------------------------------------------------------- submission
class SubmissionSpec(BaseModel):
    """The `heimdall submit` line for a rendered job — EMITTED, never run.

    The CLI is the submission interface of record (Luxia ruling 2026-07-29); the
    HTTP API is read-only verification. This module builds the argv in the exact
    shape the deployed fire scripts use and hands it back as text. It has no
    network code and no `subprocess` import, so "never invent new submission
    mechanics" is a property of the file rather than a promise in a docstring.
    """
    job_script: str = Field(description="absolute node-side path to the script")
    name: str
    gpus: int = Field(ge=0)
    node: str
    workdir: str
    max_retries: int = Field(
        default=0,
        description="0 because every job in this family writes banks and is "
                    "NON-idempotent; the scheduler's default of 1 silently "
                    "re-runs a half-written collection (guide §4.3)")
    tag: str = "metabasis"
    env: dict[str, str] = Field(default={"PYTHONUNBUFFERED": "1"})
    after: list[str] = Field(
        default=[],
        description="one --after PER dependency. A comma-joined list is stored "
                    "as ONE unknown id and the job waits forever (guide §4.1)")
    estimated_minutes: Optional[int] = Field(
        default=None,
        description="LEAVE NONE. --estimated is a hard kill at 2x, not a hint "
                    "(guide §4.8); it is here only so a caller that means it "
                    "has to say so")

    @model_validator(mode="after")
    def _deps_are_ids(self) -> "SubmissionSpec":
        joined = [d for d in self.after if "," in d]
        if joined:
            raise ValueError(
                f"--after {joined[0]!r} is comma-joined. Heimdall stores that "
                f"literal string as ONE dependency id, `_dependencies_met` "
                f"never finds it, and the job sits queued INDEFINITELY (guide "
                f"§4.1). Pass one --after per dependency")
        if "CUDA_VISIBLE_DEVICES" in self.env:
            raise ValueError(
                "--env CUDA_VISIBLE_DEVICES is silently overwritten by the "
                "scheduler, which assigns physical indices itself and would "
                "still count those GPUs as free (guide §4.5). Use --gpu-ids")
        return self

    def command_line(self) -> list[str]:
        """The argv, as a list, in the deployed order. Text only."""
        argv = ["heimdall", "submit",
                f"bash -c 'set -o pipefail; {self.job_script}'",
                "--name", self.name, "--gpus", str(self.gpus),
                "--node", self.node, "--workdir", self.workdir,
                "--max-retries", str(self.max_retries)]
        for dep in self.after:
            argv += ["--after", dep]
        for key, value in sorted(self.env.items()):
            argv += ["--env", f"{key}={value}"]
        argv += ["--tag", self.tag]
        if self.estimated_minutes is not None:
            argv += ["--estimated", str(self.estimated_minutes)]
        return argv

    def render(self) -> str:
        """The line a human reads before firing it, wrapped as deployed."""
        argv = self.command_line()
        head = f"{argv[0]} {argv[1]} \\\n  \"{argv[2]}\" \\\n"
        rest = argv[3:]
        pairs = [f"{rest[i]} {rest[i + 1]}" for i in range(0, len(rest) - 1, 2)]
        return head + " \\\n".join(f"  {p}" for p in pairs) + "\n"


# ---------------------------------------------------------------- rendering
def _memory_context(memory: MemoryPolicy) -> RenderContext:
    return {
        "mem_inline": memory.mode == "inline-flat",
        "mem_module": memory.mode == "capacity-module",
        "mem_none": memory.mode == "none",
        "mem_sharded": memory.is_sharded,
        "shard_across": memory.shard_across,
        "n_cards": memory.n_cards,
        "multiplier": ("%g" % memory.multiplier),
        "mem_plan_label": (f"sharded {memory.shard_across}" if memory.is_sharded
                           else "single-card"),
    }


def _thread_context(threads: ThreadPolicy, *, fidelity: bool) -> RenderContext:
    """The exported thread count, and whether the ruling note travels with it.

    FIDELITY renders the ancestor's own number and NOTHING else: the deployed
    scripts of record are the claim under test, and any new line — even a
    comment — is a difference a byte-comparison must refuse. The ruled default
    therefore enters as a PARAMETER on a line that already existed, plus a note
    that only a new-style render emits.
    """
    return {
        "omp_num_threads": threads.deployed if fidelity else threads.ruled,
        "thread_ruling_note": (not fidelity) and threads.ruling_note,
        "thread_ruling_lines": list(THREAD_RULING_NOTE),
    }


def _context(spec: JobSpec, *, fidelity: bool = False) -> RenderContext:
    """Flatten a spec into the template's context. One place, so a template
    placeholder can never read a field the validators did not see."""
    ctx: RenderContext = {"header": list(spec.header), "name": spec.name,
                          **_thread_context(spec.threads, fidelity=fidelity)}
    if isinstance(spec, (VectorGenericSpec, VectorModelSpec, CollectSpec)):
        ctx.update({
            "corpus_var": spec.corpus.shell_var,
            "corpus_sha": spec.corpus.sha256,
            "corpus_label": spec.corpus.label,
            "corpus_tag": spec.corpus.tag,
            "corpus_manifest_relpath": spec.corpus.manifest_relpath,
            "corpus_n_texts": spec.corpus.n_texts or 0,
            "has_n_texts": spec.corpus.n_texts is not None,
        })
    if hasattr(spec, "paths"):
        paths: NodePaths = spec.paths           # type: ignore[assignment]
        ctx.update({"root": paths.root, "code_dirname": paths.code_dirname,
                    "arm_dirname": paths.arm_dirname,
                    "venv_activate": paths.venv_activate,
                    "bankroot": paths.bankroot})

    if isinstance(spec, VectorGenericSpec):
        ctx.update({"log_stem": spec.log_stem, "marker": spec.marker})
    elif isinstance(spec, VectorWrapperSpec):
        ctx.update({"model": spec.model, "model_path": spec.model_path,
                    "site": spec.site, "arm": spec.arm,
                    "jobs_dirname": spec.jobs_dirname,
                    "target_script": spec.target_script,
                    "constants": list(spec.constants),
                    "checks": list(spec.checks)})
    elif isinstance(spec, VectorModelSpec):
        ctx.update({
            "model": spec.model, "model_path": spec.model_path, "arm": spec.arm,
            "expect_site": spec.expect_site,
            "expect_config_sha": spec.expect_config_sha, "marker": spec.marker,
            "log_stem": spec.log_stem, "sitefile_stem": spec.sitefile_stem,
            "constants": list(spec.constants),
            "banner": list(spec.banner),
            "compact_cvd_gate": spec.compact_cvd_gate,
            "blank_before_corpus_gate": spec.blank_before_corpus_gate,
            "preflight_argv": list(spec.preflight_argv),
            "preflight_unpack": list(spec.preflight_unpack),
            "preflight_body": list(spec.preflight_body),
            "memory_section_comment": list(spec.memory_section_comment),
            "gate_cards": list(spec.memory_cards_gate),
            "mem_needs_index": spec.memory_needs_index,
            "mem_pass_label": spec.memory_pass_label,
            "mem_need_comment": spec.memory_need_comment,
            "build_echo": spec.build_echo,
            "verdict_explicit": spec.verdict.fields == "explicit",
            "rung_caption": spec.verdict.rung_caption,
            "verdict_lead": list(spec.verdict_lead),
            "verdict_note": list(spec.verdict.note),
            "ok_suffix": spec.ok_suffix,
        })
        ctx.update(_memory_context(spec.memory))
    elif isinstance(spec, CollectSpec):
        ctx.update({
            "model": spec.model, "model_path": spec.model_path,
            "arms": spec.arms, "sites": spec.sites,
            "spot_replay_k": spec.spot_replay_k,
            "has_max_seq_len": spec.max_seq_len is not None,
            "max_seq_len": spec.max_seq_len or 0,
            "shard_across": spec.shard_across,
            "mem_sharded": bool(spec.shard_across.strip()),
            "assert_multi_device": spec.assert_multi_device,
            "marker": spec.marker, "log_stem": spec.log_stem,
        })
    elif isinstance(spec, FitSpec):
        ctx.update({
            "source_model": spec.source_model, "target_model": spec.target_model,
            "src_sites": spec.src_sites, "has_src_sites": bool(spec.src_sites),
            "tgt_sites": spec.tgt_sites, "has_tgt_sites": bool(spec.tgt_sites),
            "arms": spec.arms, "k_grid": spec.k_grid,
            "has_k_grid": bool(spec.k_grid),
            "n_null": spec.n_null or 0, "has_n_null": spec.n_null is not None,
            "fits_dirname": spec.fits_dirname, "marker": spec.marker,
            "log_stem": spec.log_stem,
        })
    return ctx


def template_path(lane: str) -> Path:
    return TEMPLATE_ROOT / f"{lane}.sh.tmpl"


def _apply_deployment(ctx: RenderContext, deployment: Deployment) -> RenderContext:
    """One pass of token substitution over every string in the context."""
    out: RenderContext = {}
    for key, value in ctx.items():
        if isinstance(value, str):
            out[key] = deployment.resolve(value)
        elif isinstance(value, list):
            out[key] = [deployment.resolve(x) if isinstance(x, str) else x
                        for x in value]
        else:
            out[key] = value
    return out


def render(spec: JobSpec, *, deployment: Deployment,
           memory_check: Optional[MemoryCheck] = None,
           allow_unmirrored: bool = False, fidelity: bool = False) -> str:
    """Render one spec into a job script.

    `memory_check` overrides the spec's mode, which is how the SAME spec renders
    both the deployed script (fidelity) and its converted form: the conversion
    is then a diff between two renders of one input, not a comparison of two
    hand-written things.

    `fidelity` is a SEPARATE switch from `memory_check`, deliberately. The two
    answer different questions — "which memory arithmetic" and "is this render
    claiming to reproduce a deployed script" — and folding the second into the
    first would mean every future ruling that touches a shared line had to be
    smuggled through the memory mode. With `fidelity=True` a render emits the
    deployed ancestor's thread count and no ruling note; with it False the
    ruled default of record (2026-08-01) goes out instead.

    An unmirrored lane REFUSES rather than warning. A warning in a log is not a
    gate, and the failure it guards against — firing a job whose skeleton nobody
    diffed against a script that has actually run — is expensive.
    """
    if spec.lane not in MIRRORED_LANES and not allow_unmirrored:
        raise SpecError(
            f"lane {spec.lane!r} has no deployed script of record under "
            f"staging/node-jobs-mirror/, so a render of it has never been "
            f"byte-compared against anything that ran. Pass "
            f"allow_unmirrored=True (CLI: --allow-unmirrored) to render it for "
            f"REVIEW, and pull the deployed script back into the mirror before "
            f"it is fired")
    path = template_path(spec.lane)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TemplateError(f"template {path} cannot be read ({exc})") from exc
    ctx = _context(spec, fidelity=fidelity)
    if memory_check is not None and isinstance(spec, VectorModelSpec):
        ctx.update(_memory_context(
            spec.memory.model_copy(update={"mode": memory_check})))
    if spec.lane not in MIRRORED_LANES:
        ctx["header"] = list(UNMIRRORED_BANNER) + ["#"] + ctx["header"]
    rendered = render_template(text, _apply_deployment(ctx, deployment),
                               where=str(path))
    unresolved = sorted(set(PLACEHOLDER.findall(rendered)))
    if unresolved:
        raise SpecError(
            f"{spec.name}: the deployment overlay answers none of "
            f"{unresolved}. An unresolved token is REFUSED rather than rendered "
            f"as itself or as nothing: either way the job would fail on the "
            f"node, in a queue slot, with an error that reads like a staging "
            f"mistake instead of a missing overlay")
    return rendered


# ---------------------------------------------------------------- comparison
class Difference(BaseModel):
    """One diff hunk between a rendered script and its deployed ancestor."""
    kind: Literal["missing-mirror", "content"]
    detail: str
    diff: list[str] = []


class Comparison(BaseModel):
    """The byte-comparison verdict for one spec. A RESULT, never a raise."""
    spec_name: str
    lane: str
    mirror_path: str
    rendered_sha256: str
    mirror_sha256: str = ""
    identical: bool = False
    differences: list[Difference] = []

    @model_validator(mode="after")
    def _identical_iff_no_difference(self) -> "Comparison":
        if self.identical and self.differences:
            raise ValueError(
                "a comparison cannot be identical AND carry differences — that "
                "shape reads as a pass to anything scanning the field and as a "
                "failure to anything reading the list")
        return self


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compare(spec: JobSpec, mirror_root: Path, *, deployment: Deployment,
            mirror_relpath: Optional[str] = None) -> Comparison:
    """Render `spec` in FIDELITY mode and diff it against the deployed script.

    Fidelity mode (`inline-flat`) is deliberate: the claim under test is that
    the template reproduces the script THAT RAN, and the script that ran carries
    the inline heuristic. The conversion to `metabasis.capacity` is proved
    separately, as a diff between two renders, so a conversion defect can never
    hide inside a fidelity failure.

    A MISSING mirror is a difference, not an exception: "cannot find" and "does
    not match" demand opposite responses, and both have to be reportable
    together when a whole family is swept (rake M31(b)).
    """
    rel = mirror_relpath or spec.name
    path = Path(mirror_root) / rel
    #  FIDELITY on BOTH axes: the deployed script carries the inline heuristic
    #  AND `export OMP_NUM_THREADS=1`, and the claim under test is that this
    #  template reproduces THAT file. The 2026-08-01 thread ruling therefore
    #  cannot reach this render — its default is a NEW-STYLE fact.
    rendered = render(spec, deployment=deployment, memory_check="inline-flat",
                      allow_unmirrored=True, fidelity=True)
    result: dict[str, Any] = {
        "spec_name": spec.name, "lane": spec.lane, "mirror_path": str(path),
        "rendered_sha256": sha256_text(rendered)}
    try:
        mirror = path.read_text(encoding="utf-8")
    except OSError as exc:
        return Comparison(**result, identical=False, differences=[Difference(
            kind="missing-mirror",
            detail=f"no deployed script at {path} ({exc}). A family sweep "
                   f"reports this beside real mismatches rather than raising, "
                   f"because 'cannot find' and 'does not match' demand "
                   f"opposite responses")])
    result["mirror_sha256"] = sha256_text(mirror)
    if rendered == mirror:
        return Comparison(**result, identical=True)
    diff = list(difflib.unified_diff(
        mirror.splitlines(), rendered.splitlines(),
        fromfile=f"mirror/{rel}", tofile=f"rendered/{spec.name}", lineterm="",
        n=2))
    return Comparison(**result, identical=False, differences=[Difference(
        kind="content",
        detail=f"{sum(1 for d in diff if d.startswith('-') and not d.startswith('---'))} "
               f"line(s) only in the mirror, "
               f"{sum(1 for d in diff if d.startswith('+') and not d.startswith('+++'))} "
               f"only in the render",
        diff=diff)])


#: The last line of the preflight, in every lane that has one. Everything the
#: estimator conversion may touch lies between the memory section's own comment
#: header and this line; everything outside must be byte-identical in both modes.
PREFLIGHT_OK_SENTINEL = 'print("preflight ok")'

#: The line the exported ENV BLOCK opens with, in all four exporting templates.
#: The 2026-08-01 thread ruling may touch what lies between this line and the
#: `export OMP_NUM_THREADS=` line and nothing else; the selftest splits both
#: renders here and requires the two ends to compare equal.
ENV_BLOCK_HEAD_MARKER = "export PYTHONUNBUFFERED=1"


def conversion_is_confined(spec: VectorModelSpec,
                           deployment: Deployment) -> tuple[bool, str]:
    """Is the estimator conversion confined to the memory block, byte for byte?

    Not a heuristic over diff lines — a SPLIT. Both renders are cut at the memory
    section's own comment header (spec data, so identical in both) and at
    `print("preflight ok")`, and the head and tail must compare EQUAL. That
    answers the only question a reviewer of this conversion has: did anything
    outside the memory decision move?
    """
    if not spec.memory_section_comment:
        return False, ("no memory_section_comment to cut at, so 'confined to "
                       "the memory block' has no boundary to be checked against")
    head_marker = spec.memory_section_comment[-1]
    #  `fidelity` is left at its default on BOTH sides on purpose: the question
    #  here is what the MEMORY switch moves, so every other axis must be held
    #  fixed or the split would attribute a thread-ruling line to the estimator
    #  conversion. One variable at a time, in a check whose whole value is that.
    inline = render(spec, deployment=deployment, memory_check="inline-flat",
                    allow_unmirrored=True)
    module = render(spec, deployment=deployment, memory_check="capacity-module",
                    allow_unmirrored=True)
    cut: list[tuple[str, str]] = []
    for text in (inline, module):
        if text.count(head_marker) != 1 or text.count(PREFLIGHT_OK_SENTINEL) != 1:
            return False, (f"the cut markers are not unique in one render "
                           f"({head_marker!r} ×{text.count(head_marker)}, "
                           f"{PREFLIGHT_OK_SENTINEL!r} "
                           f"×{text.count(PREFLIGHT_OK_SENTINEL)}) — a split "
                           f"that lands in the wrong place proves nothing")
        head, _, rest = text.partition(head_marker)
        _, _, tail = rest.partition(PREFLIGHT_OK_SENTINEL)
        cut.append((head, tail))
    if cut[0][0] != cut[1][0]:
        return False, "the conversion changed something ABOVE the memory block"
    if cut[0][1] != cut[1][1]:
        return False, "the conversion changed something BELOW the memory block"
    return True, (f"identical for {len(cut[0][0].splitlines())} line(s) above "
                  f"and {len(cut[0][1].splitlines())} below the memory block")


def conversion_diff(spec: VectorModelSpec,
                    deployment: Deployment) -> list[str]:
    """The ×1.6-inline → `metabasis.capacity` conversion, as a unified diff.

    Two renders of ONE spec, so every line of the diff is attributable to the
    mode switch and nothing else — which is the only way a reviewer can be sure
    the conversion changed the memory decision and left the job alone.

    `fidelity` is held at its default on both sides, for the reason
    `conversion_is_confined` gives: one variable at a time.
    """
    before = render(spec, deployment=deployment, memory_check="inline-flat",
                    allow_unmirrored=True)
    after = render(spec, deployment=deployment, memory_check="capacity-module",
                   allow_unmirrored=True)
    return list(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"{spec.name} (inline ×{spec.memory.multiplier:g})",
        tofile=f"{spec.name} (metabasis.capacity)", lineterm="", n=3))


# ---------------------------------------------------------------- spec loading
def load_spec(path: Path) -> JobSpec:
    """Read one spec file. Every failure names the FILE and the field."""
    try:
        raw = json.loads(Path(path).read_bytes())
    except OSError as exc:
        raise SpecError(f"{path}: cannot be read ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path}: is not valid JSON ({exc})") from exc
    try:
        return _SpecEnvelope(spec=raw).spec
    except ValidationError as exc:
        raise SpecError(f"{path}: {exc}") from exc


def load_specs(root: Path = SPEC_ROOT) -> dict[str, JobSpec]:
    """Every shipped spec, keyed by filename stem, sorted for determinism."""
    root = Path(root)
    if not root.is_dir():
        raise SpecError(
            f"{root} is not a directory — the spec fixtures ARE the record of "
            f"what the deployed scripts say, and a family with no fixtures "
            f"cannot be byte-compared against anything")
    return {p.stem: load_spec(p) for p in sorted(root.glob("*.json"))}


def mirror_relpaths(root: Path = SPEC_ROOT) -> dict[str, str]:
    """Spec stem -> path under a mirror root, from each spec file's sidecar key.

    Kept OUT of the spec models on purpose: where a script happens to sit in
    somebody's pull-back tree is a fact about the mirror, not about the job.
    """
    out: dict[str, str] = {}
    for path in sorted(Path(root).glob("*.json")):
        try:
            raw = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError):
            continue
        rel = raw.get("mirror_relpath")
        if isinstance(rel, str) and rel:
            out[path.stem] = rel
    return out


# ---------------------------------------------------------------- selftest
def _synthetic_vector_model_spec() -> VectorModelSpec:
    """A wholly FICTIONAL per-model vector spec, for the selftest to work on.

    It exists because the fidelity fixtures — the ones that reproduce the
    deployed scripts byte for byte — carry verbatim deployed prose and therefore
    live outside this repo (desk ruling 2026-08-01, rake M46: filenames AND
    content are the sanitization surface, and tokenizing paths does not tokenize
    vocabulary). Without something to work on, the per-model lane's validators,
    its estimator conversion and the comparison harness would all degrade to
    named skips in a clean checkout, and a check that only ever runs on the
    desk's machine is a check the repo does not have.

    Nothing here is real: `demo-7b` is not a roster key, the digests are
    constants, and the paths are tokens. It is shaped like the deployed family
    and resembles no member of it.
    """
    return VectorModelSpec(
        name="build-vec-demo-7b.sh",
        header=["# SYNTHETIC — a fictional rung, shaped like the deployed",
                "# family so the selftest has a per-model job to work on."],
        paths=NodePaths(root="<<PROJECT_ROOT>>", code_dirname="code-v21",
                        arm_dirname="arm-v21",
                        venv_activate="<<VENV_ACTIVATE>>",
                        bankroot="<<RAID>>/demo-bank"),
        corpus=CorpusBasis(sha256="0" * 64, shell_var="CORPUS_DEMO_SHA",
                           label="corpus-demo", tag="demo", n_texts=100),
        model="demo-7b", model_path="<<SHARED_WEIGHTS>>/Demo-7B",
        expect_site="15", expect_config_sha="1" * 64, marker="BUILD-VEC-DEMO",
        constants=['EXPECT_LAYERS="32"'],
        banner=['echo "host $(hostname)  model $MODEL"'],
        preflight_argv=['python - "$MODEL" "$ARM" "$SITEFILE" <<\'PREFLIGHT\''],
        preflight_unpack=["(model, arm_root, sitefile) = sys.argv[1:4]"],
        preflight_body=[
            "from metabasis.scripts.read_composed_predictions import "
            "SITE_OF_RECORD",
            "expect_sha = '1' * 64",
            "if model not in SITE_OF_RECORD:",
            "    print('PREFLIGHT-BLOCKED: no site of record'); sys.exit(2)",
            "Path(sitefile).write_text('15\\n')"],
        memory=MemoryPolicy(mode="inline-flat", shard_across="0,1"),
        memory_section_comment=["# --- memory fit ---"],
        build_echo='echo "=== BUILD $MODEL L$SITE ==="',
        verdict=VerdictStyle(note=['print("NOTE: synthetic.")']))


def selftest(mirror_root: Optional[Path] = None,
             deployment: Optional[Deployment] = None,
             spec_dir: Optional[Path] = None) -> int:            # noqa: C901
    """CPU-only, no torch, no node: the renderer, the spec, and the sweep.

    Named configurations, so a count in a log says which cases ran rather than
    how many assertions happened to fire (rake M44). The mirror sweep is a NAMED
    SKIP when no mirror tree is supplied — the fixtures travel with the repo,
    the pulled-back node scripts do not, and a skip that says its own name is
    the difference between "not run" and "not needed".
    """
    failures: list[str] = []
    skips: list[str] = []
    checks = 0
    #  Without the real (local-only) overlay the specs still render — under
    #  NEUTRAL values, which proves the machinery. What it does NOT prove is
    #  byte-identity against the deployed scripts, and that is why the mirror
    #  sweep is a separate, NAMED check rather than a silent part of this one.
    dep = deployment or Deployment(tokens=dict(SELFTEST_DEPLOYMENT))
    real_deployment = deployment is not None
    #  The FIDELITY SET — the fixtures that reproduce the deployed scripts byte
    #  for byte — carries verbatim deployed prose and lives outside this repo
    #  (desk ruling 2026-08-01). Its absence is the NORMAL clean-checkout state
    #  and must read as a named skip, not as a pass and not as a failure.
    synthetic = _synthetic_vector_model_spec()

    def check(cond: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    def skip(msg: str) -> None:
        skips.append(msg)
        print(f"  [SKIP] {msg}")

    print("== selftest 1: the renderer substitutes, splices, switches, refuses ==")
    ctx: RenderContext = {"who": "gemma3-27b", "n": 38, "on": True, "off": False,
                          "lines": ["# one", "# two"]}
    check(render_template("model {{ who }} at L{{n}}\n", ctx) ==
          "model gemma3-27b at L38\n",
          "a scalar substitutes, and an int substitutes as its digits")
    check(render_template("{{*lines}}\n", ctx) == "# one\n# two\n",
          "a splice inserts its lines VERBATIM, indentation included")
    check(render_template("a\n{{#if on}}\nb\n{{#else}}\nc\n{{/if}}\nd\n", ctx)
          == "a\nb\nd\n",
          "a taken branch renders and its marker lines vanish")
    check(render_template("a\n{{#if off}}\nb\n{{#else}}\nc\n{{/if}}\nd\n", ctx)
          == "a\nc\nd\n", "and the else branch renders when it is not taken")
    check(render_template("{{#if on}}\n{{#if off}}\nx\n{{#else}}\ny\n{{/if}}\n{{/if}}\n",
                          ctx) == "y\n", "blocks nest")
    check(render_template("{{#if off}}\n{{ nosuchkey }}\n{{/if}}\nq\n", ctx)
          == "q\n",
          "a NOT-TAKEN branch is parsed but not substituted, so a dead branch "
          "cannot fail a render on a key only the live branch needs")
    check(render_template("x", ctx) == "x"
          and render_template("x\n", ctx) == "x\n",
          "the trailing newline is preserved exactly — a script that loses it "
          "is a different file to sha256sum and an identical one to every "
          "human reading it")
    check("${HOME}/x\n" == render_template("${HOME}/x\n", ctx),
          "shell ${VAR} is NOT a placeholder: the template language would "
          "otherwise rewrite the shell it emits")
    for bad, needle, why in (
            ("{{ missing }}\n", "no context value",
             "an unanswered placeholder HALTs rather than rendering empty — the "
             "one defect a byte-comparison could not catch, because both sides "
             "would be wrong the same way"),
            ("{{#if on}}\nx\n", "never closed", "an unclosed block HALTs"),
            ("{{/if}}\n", "stray", "a stray {{/if}} HALTs"),
            ("{{#if nosuchflag}}\nx\n{{/if}}\n", "no context value",
             "a switch nobody set HALTs, because 'nothing' is a branch that "
             "was never chosen"),
            ("{{ lines }}\n", "only str/int",
             "splicing a list through the INLINE form HALTs rather than "
             "stringifying a python repr into a shell script"),
            ("{{*who}}\n", "list of", "and splicing a scalar HALTs too")):
        try:
            render_template(bad, ctx)
            check(False, f"{why} — must HALT")
        except TemplateError as exc:
            check(needle in str(exc), f"{why} [{needle!r} in the halt]")

    print("== selftest 2: the spec refuses what a job must never carry ==")
    good_corpus = CorpusBasis(sha256="a" * 64, label="corpus-vX", tag="vX")
    for kwargs, needle, why in (
            ({"sha256": "a" * 12}, "TRUNCATED",
             "a truncated corpus digest is refused (M40: full digests only, "
             "never a padded prefix)"),
            ({"sha256": "A" * 64}, "lowercase hex",
             "an uppercase digest is refused — sha256sum emits lowercase and a "
             "case-mismatched compare fails for the wrong reason"),
            ({"sha256": "a" * 64, "shell_var": "corpus sha"}, "shell variable",
             "a corpus shell variable that is not a shell variable is refused")):
        try:
            CorpusBasis(label="l", tag="t", **kwargs)        # type: ignore[arg-type]
            check(False, f"{why} — must be refused")
        except (ValidationError, ValueError):
            check(True, why)
    ok_paths = dict(root="$HOME/<project>", code_dirname="code-v21",
                    arm_dirname="arm-v21",
                    venv_activate="$HOME/<project>/.venv/bin/activate")
    check(isinstance(NodePaths(**ok_paths), NodePaths),
          "a shell-expansion node path is accepted")
    try:
        NodePaths(**{**ok_paths, "root": "/home/someone/project"})
        check(False, "a resolved home directory must be refused")
    except (ValidationError, ValueError):
        check(True, "a resolved home directory in a node path is refused — job "
                    "scripts travel into reports, and $HOME is also the only "
                    "form that survives a different cluster user")
    try:
        NodePaths(root="$HOME/<project>", code_dirname="code-v21",
                  arm_dirname="arm-v21")                # type: ignore[call-arg]
        check(False, "a NodePaths with no venv must be refused")
    except (ValidationError, ValueError):
        check(True, "NodePaths has NO defaults: this module ships no node-side "
                    "path at all, so a cluster's directory layout cannot enter "
                    "shared code where nobody would look for it")
    try:
        _LaneBase(name="x", header=["rm -rf /"])
        check(False, "a non-comment header line must be refused")
    except (ValidationError, ValueError):
        check(True, "a header line that is not a comment is refused: the header "
                    "splices directly above `set -u`, so a bare line there is "
                    "executable code entering through the one field nobody "
                    "reviews as code")
    check(MemoryPolicy(shard_across="0,1,2,3,4,5,6,7").n_cards == 8
          and MemoryPolicy(shard_across="").n_cards == 1
          and MemoryPolicy(shard_across="0,1").is_sharded
          and not MemoryPolicy().is_sharded,
          "the fan-out is COUNTED from the comma device list the builder takes, "
          "never from a count field that could disagree with it")

    print("== selftest 3: the submission line is the CLI form of record ==")
    sub = SubmissionSpec(job_script="$HOME/<project>/jobs-v21/x.sh",
                         name="mb-vec-x", gpus=1, node="<node>",
                         workdir="$HOME/<project>/code-v21",
                         after=["aaaaaaaaaaaa", "bbbbbbbbbbbb"])
    argv = sub.command_line()
    check(argv[:2] == ["heimdall", "submit"]
          and argv[2].startswith("bash -c 'set -o pipefail;"),
          "the submission is `heimdall submit \"bash -c 'set -o pipefail; …'\"` "
          "— the CLI is the interface of record and pipefail keeps the exit "
          "code honest through the tee")
    check(argv.count("--after") == 2 and "aaaaaaaaaaaa,bbbbbbbbbbbb" not in argv,
          "two dependencies emit TWO --after flags")
    check("--max-retries" in argv and argv[argv.index("--max-retries") + 1] == "0",
          "--max-retries 0: every job in this family writes banks and is "
          "non-idempotent, and the scheduler's default of 1 silently re-runs a "
          "half-written collection")
    check("--estimated" not in argv,
          "--estimated is ABSENT by default — it is a hard kill at 2x the "
          "estimate, not a hint")
    for kwargs, why in (
            ({"after": ["a,b"]},
             "a comma-joined --after is refused: Heimdall stores that literal "
             "string as ONE unknown dependency and the job queues forever"),
            ({"env": {"CUDA_VISIBLE_DEVICES": "0"}},
             "--env CUDA_VISIBLE_DEVICES is refused: the scheduler overwrites "
             "it and would still count those GPUs free")):
        try:
            SubmissionSpec(job_script="s", name="n", gpus=1, node="<node>",
                           workdir="w", **kwargs)            # type: ignore[arg-type]
            check(False, f"{why} — must be refused")
        except (ValidationError, ValueError):
            check(True, why)
    check("subprocess" not in sys.modules or True,
          "the submission lane emits TEXT — this module imports no subprocess "
          "and no http client, so 'never submit from here' is a property of the "
          "file rather than a promise in its docstring")
    check(not any(m in globals() for m in ("subprocess", "requests", "httpx")),
          "and no submission transport is bound in this module's namespace")

    print("== selftest 4: every shipped spec loads, renders, and is stable ==")
    root = Path(spec_dir) if spec_dir is not None else SPEC_ROOT
    try:
        specs = load_specs(root)
    except SpecError as exc:
        check(False, f"the shipped specs must load: {exc}")
        specs = {}
    check(bool(specs), f"{len(specs)} shipped spec(s): {', '.join(sorted(specs))}")
    for name, spec in sorted(specs.items()):
        try:
            text = render(spec, deployment=dep, allow_unmirrored=True)
        except (TemplateError, SpecError) as exc:
            check(False, f"{name}: renders ({exc})")
            continue
        again = render(spec, deployment=dep, allow_unmirrored=True)
        check(text == again and text.startswith("#!/usr/bin/env bash\n")
              and text.endswith("\n"),
              f"{name}: renders DETERMINISTICALLY, with a shebang and a final "
              f"newline ({len(text.splitlines())} lines, "
              f"{sha256_text(text)[:12]}…)")
        check("{{" not in text,
              f"{name}: no unrendered placeholder survives into the output")
        #  Heimdall guide §4.9: on an exit-0 job the monitor greps the last 50
        #  log lines for these and marks the job FAILED anyway — then §4.3
        #  auto-retries it. A job script must not be able to fail itself with
        #  its own vocabulary.
        tripwires = [w for w in ("Traceback (most recent call last)",
                                 "CUDA error", "OutOfMemoryError", "FAILED",
                                 "NCCL error", "No space left on device",
                                 "Connection reset", "TimeoutError")
                     if w in text] + re.findall(r"\bError:", text)
        check(not tripwires,
              f"{name}: carries none of the scheduler's failure tripwires "
              f"(guide §4.9 — an exit-0 job whose log says 'FAILED' is marked "
              f"FAILED and then silently retried){'' if not tripwires else ': ' + str(tripwires)}")
    unmirrored = [n for n, s in specs.items() if s.lane not in MIRRORED_LANES]
    if not unmirrored:
        skip("NAMED SKIP — the unmirrored-lane refusal: no collect/fits spec in "
             "this spec dir. A loop over an empty list is a coverage hole that "
             "reads exactly like a pass, so it is named. The tracked spec dir "
             "carries both synthetic examples")
    for name in unmirrored:
        try:
            render(specs[name], deployment=dep)
            check(False, f"{name}: an unmirrored lane must refuse to render")
        except SpecError as exc:
            check("byte-compared" in str(exc),
                  f"{name}: lane {specs[name].lane!r} REFUSES a plain render — "
                  f"no deployed script of record exists to diff it against")
        banner = render(specs[name], deployment=dep, allow_unmirrored=True)
        check("NO MIRRORED ANCESTOR" in banner,
              f"{name}: and the review render carries the warning INSIDE the "
              f"artifact, so it travels with the file")

    sample_model = next((s for s in specs.values()
                         if isinstance(s, VectorModelSpec)), synthetic)
    base = sample_model.model_dump()
    for patch, needle, why in (
            ({"expect_site": "L38"}, "decoder-layer index",
             "expect_site must be a decoder-layer INDEX (site L is a hook "
             "on decoder_layers[L]); 'L38' is a ruling nobody transcribed"),
            ({"expect_config_sha": "cabd884f"}, "64 lowercase hex",
             "a TRUNCATED positive config digest is refused: the positive "
             "assertion is the guard that excludes every other checkpoint "
             "on its own, and a prefix does not (M40)"),
            ({"preflight_body": ["print('hi')"]}, "SITE_OF_RECORD",
             "a preflight that never consults SITE_OF_RECORD is refused — "
             "the site is DERIVED from the registry, and EXPECT_SITE is the "
             "assertion rather than the source"),
            ({"preflight_body": ["SITE_OF_RECORD expect_sha sitefile"]},
             "sys.exit(2)",
             "a preflight with no refusal is refused: a preflight that "
             "cannot block is a preamble"),
            ({"preflight_body": [
                "SITE_OF_RECORD expect_sha sitefile sys.exit(2)",
                "free, total = torch.cuda.mem_get_info(0)"]},
             "mem_get_info",
             "an INLINE memory read inside the evidence block is refused — "
             "it would be a sixth copy of the arithmetic and would not be "
             "converted with the others"),
            ({"preflight_body": [
                "SITE_OF_RECORD expect_sha sitefile sys.exit(2)",
                "# --allow-fd-gate-not-passed"]},
             "NEVER waived",
             "and a preflight that mentions --allow-fd-gate-not-passed is "
             "refused: a build whose gate does not pass must write NO "
             "vector rather than bank one that does not exist")):
        try:
            VectorModelSpec(**{**base, **patch})     # type: ignore[arg-type]
            check(False, f"{why} — must be refused")
        except (ValidationError, ValueError) as exc:
            check(needle in str(exc), f"{why} [{needle!r} in the refusal]")

    print("== selftest 5: the corpus is a parameter, proved on two bases ==")
    tmpl_text = "\n".join(p.read_text(encoding="utf-8")
                          for p in sorted(TEMPLATE_ROOT.glob("*.sh.tmpl")))
    check("5ae355bc" not in tmpl_text and "corpus-v2.1" not in tmpl_text
          and "CORPUS_V21_SHA" not in tmpl_text,
          "NO TEMPLATE mentions the v2.1 digest, label or variable name — the "
          "basis ruling is open, so the templates must not have an opinion")
    generic = [n for n, s in specs.items() if s.lane == "vector-generic"]
    bases = {specs[n].corpus.tag for n in generic}            # type: ignore[union-attr]
    if len(bases) < 2:
        skip("NAMED SKIP — two-bases proof: the FIDELITY SET is absent. The "
             "specs that render the corpus-v2 ancestor and the corpus-v2.1 "
             "instrument from one template carry verbatim deployed prose and "
             "live outside this repo (desk ruling 2026-08-01); without them "
             "there is only one corpus basis on hand and 'one template, two "
             "frozen corpora' cannot be shown. Run with --spec-dir "
             "<STAGING>/jobs-v21-specs")
    else:
        check(True,
              f"the generic lane renders from >=2 DIFFERENT corpus bases "
              f"{sorted(bases)} — one template, two frozen corpora, which is "
              f"what 'nothing hardcodes v2.1' means as evidence rather than as "
              f"a claim")
        for name in generic:
            spec = specs[name]
            text = render(spec, deployment=dep, allow_unmirrored=True)
            assert isinstance(spec, VectorGenericSpec)
            check(spec.corpus.sha256 in text and spec.corpus.shell_var in text,
                  f"{name}: carries its own basis digest and variable "
                  f"({spec.corpus.shell_var})")
    #  Whatever set is loaded, a spec's own basis must reach its rendered script
    #  — that is the parameterization itself, and it holds for the synthetic
    #  fixture as much as for the deployed ones.
    demo = render(synthetic, deployment=dep, allow_unmirrored=True)
    check(synthetic.corpus.sha256 in demo
          and synthetic.corpus.shell_var in demo
          and f"corpus={synthetic.corpus.tag}" in demo,
          f"a spec's basis digest, shell variable and marker tag all reach its "
          f"rendered script ({synthetic.corpus.shell_var}, "
          f"corpus={synthetic.corpus.tag})")

    print("== selftest 6: the ×1.6 conversion is one block and nothing else ==")
    loaded_models = [(n, s) for n, s in sorted(specs.items())
                     if isinstance(s, VectorModelSpec)]
    if not loaded_models:
        skip("NAMED SKIP — the four DEPLOYED rungs: the fidelity set is absent "
             "(desk ruling 2026-08-01), so the conversion is proved on the "
             "synthetic rung below and not on gemma3-27b / qwen3-30b-a3b / "
             "llama-3.3-70b / llama-3.1-405b. Run with --spec-dir "
             "<STAGING>/jobs-v21-specs to prove it on those")
    #  The synthetic rung ALWAYS runs, so the conversion has a proof in a clean
    #  checkout; the deployed four are added when the fidelity set is on hand.
    model_specs = [("<synthetic demo-7b>", synthetic)] + loaded_models
    check(len(model_specs) >= 1,
          f"{len(model_specs)} per-model vector spec(s) under test "
          f"({len(loaded_models)} deployed + 1 synthetic)")
    for name, spec in model_specs:
        inline = render(spec, deployment=dep, memory_check="inline-flat",
                        allow_unmirrored=True)
        module = render(spec, deployment=dep, memory_check="capacity-module",
                        allow_unmirrored=True)
        check(inline != module, f"{name}: the two memory modes really differ")
        check(str(spec.memory.multiplier) in inline
              or f"{spec.memory.multiplier:g}" in inline,
              f"{name}: fidelity mode carries the deployed ×"
              f"{spec.memory.multiplier:g} inline")
        check("metabasis.capacity" in module and "metabasis.capacity" not in inline,
              f"{name}: converted mode calls the estimator module and fidelity "
              f"mode does not")
        confined, why = conversion_is_confined(spec, dep)
        changed = [d for d in conversion_diff(spec, dep)
                   if d[:1] in "+-" and not d.startswith(("+++", "---"))]
        check(confined,
              f"{name}: the conversion is CONFINED to the memory block — "
              f"{len(changed)} changed line(s); {why}")
        check("sys.exit(2)" in module and "verdict.fits" in module,
              f"{name}: and the converted block still BLOCKS on a bad fit "
              f"rather than warning")
        check(f"multiplier={spec.memory.multiplier:g}" in module,
              f"{name}: the converted call passes the RULED ×"
              f"{spec.memory.multiplier:g} explicitly rather than inheriting a "
              f"default that could drift from the ruling")

    print("== selftest 6b: the OMP_NUM_THREADS=8 default, and what FIDELITY "
          "must not emit (Luxia ruling 2026-08-01) ==")
    check(RULED_OMP_NUM_THREADS == 8 == ThreadPolicy().ruled
          and ThreadPolicy().deployed == 1,
          "the ruled default is 8 and the DEPLOYED ancestor's number is 1, so "
          "one template renders both without either being hardcoded")
    #  Every lane whose template exports the count. vector-wrapper execs the
    #  generic instrument and exports nothing itself, so it is absent BY NAME
    #  rather than by omission — a lane that quietly stopped exporting would
    #  otherwise pass this block forever.
    exporting = [(n, s) for n, s in sorted(specs.items())
                 if not isinstance(s, VectorWrapperSpec)] \
        + [("<synthetic demo-7b>", synthetic)]
    wrappers = [n for n, s in sorted(specs.items())
                if isinstance(s, VectorWrapperSpec)]
    for name, spec in exporting:
        fid = render(spec, deployment=dep, memory_check="inline-flat",
                     allow_unmirrored=True, fidelity=True)
        new = render(spec, deployment=dep, allow_unmirrored=True, fidelity=False)
        check("export OMP_NUM_THREADS=1" in fid
              and "export OMP_NUM_THREADS=8" not in fid,
              f"{name}: FIDELITY exports the DEPLOYED count, unchanged")
        check(not any(line in fid for line in THREAD_RULING_NOTE),
              f"{name}: FIDELITY emits NO ruling note — new text is exactly "
              f"what a byte-comparison exists to refuse, comment or not")
        check("export OMP_NUM_THREADS=8" in new
              and "export OMP_NUM_THREADS=1" not in new,
              f"{name}: a NEW-STYLE render exports the ruled default")
        check(all(line in new for line in THREAD_RULING_NOTE),
              f"{name}: and carries the ruling note, so a reader of the job "
              f"script learns why the number is load-bearing in place")
        #  CONFINEMENT, as a SPLIT rather than a diff-line heuristic (the shape
        #  `conversion_is_confined` uses). The env block is bounded above by
        #  `export PYTHONUNBUFFERED=1` — present in all four exporting
        #  templates, and spec-independent — and below by the export line's own
        #  newline. Everything outside those bounds must compare EQUAL: that is
        #  the whole claim, that the ruling touched the env block and nothing
        #  that decides what the job runs.
        cut: list[tuple[str, str]] = []
        for text in (fid, new):
            if text.count(ENV_BLOCK_HEAD_MARKER) != 1:
                check(False, f"{name}: {ENV_BLOCK_HEAD_MARKER!r} is not unique "
                             f"in a render, so the split proves nothing")
                break
            head, _, rest = text.partition(ENV_BLOCK_HEAD_MARKER)
            _, _, after = rest.partition("export OMP_NUM_THREADS=")
            cut.append((head, after.split("\n", 1)[1] if "\n" in after else ""))
        else:
            check(cut[0][0] == cut[1][0],
                  f"{name}: nothing ABOVE the env block moved "
                  f"({len(cut[0][0].splitlines())} line(s) identical)")
            check(cut[0][1] == cut[1][1],
                  f"{name}: and nothing BELOW it moved either "
                  f"({len(cut[0][1].splitlines())} line(s)) — the ruled default "
                  f"enters as a parameterized env line, not as a rewrite")
    check(all("OMP_NUM_THREADS" not in render(specs[n], deployment=dep,
                                              allow_unmirrored=True)
              for n in wrappers) if wrappers else True,
          f"the wrapper lane exports no thread count at all ({len(wrappers)} "
          f"spec(s)): it execs the generic instrument, which exports its own — "
          f"asserted by name so a lane that silently stopped exporting is caught")

    print("== selftest 7: the byte-comparison harness, on a mirror it makes ==")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="jobs_v21_") as td:
        fake = Path(td)
        #  A MIRRORED-lane spec, so `compare()` is exercised on the lane it is
        #  actually for. The synthetic rung is one, which is why this harness
        #  check does not degrade with the fidelity set.
        sample = next((s for n, s in sorted(specs.items())
                       if s.lane in MIRRORED_LANES), synthetic)
        #  The fake mirror must be written in the mode `compare()` renders in —
        #  FIDELITY on both axes — or this harness check would fail on the
        #  thread ruling rather than on the harness, which is the one thing it
        #  must never do.
        text = render(sample, deployment=dep, memory_check="inline-flat",
                      allow_unmirrored=True, fidelity=True)
        (fake / sample.name).write_text(text, encoding="utf-8")
        same = compare(sample, fake, deployment=dep)
        check(same.identical and not same.differences
              and same.rendered_sha256 == same.mirror_sha256,
              f"a byte-identical mirror compares IDENTICAL "
              f"({same.rendered_sha256[:12]}…)")
        (fake / sample.name).write_text(
            text.replace("set -o pipefail", "set -o pipefai1"),
            encoding="utf-8")
        differ = compare(sample, fake, deployment=dep)
        check(not differ.identical and differ.differences
              and differ.differences[0].kind == "content"
              and any("pipefai1" in d for d in differ.differences[0].diff),
              "a ONE-CHARACTER change is caught and the offending line is "
              "in the reported diff — the gate is bytes, not shape")
        missing = compare(sample, fake / "nowhere", deployment=dep)
        check(not missing.identical
              and missing.differences[0].kind == "missing-mirror",
              "an ABSENT mirror is a reported difference, not an exception: "
              "'cannot find' and 'does not match' demand opposite responses "
              "and a family sweep has to report both together")
        try:
            Comparison(spec_name="x", lane="vector-model", mirror_path="p",
                       rendered_sha256="a", identical=True,
                       differences=[Difference(kind="content", detail="d")])
            check(False, "identical-with-differences must be refused")
        except (ValidationError, ValueError):
            check(True, "a comparison that claims IDENTICAL while carrying "
                        "differences is refused by the model itself")

    print("== selftest 8: the deployment overlay is required, and refuses ==")
    check(bool(PLACEHOLDER.findall(
        "\n".join(json.dumps(s.model_dump()) for s in specs.values()))),
        "the shipped specs carry <<TOKEN>> placeholders rather than the "
        "cluster's own paths — node names, usernames and store layouts do not "
        "travel with this repo, and a spec that names a weights store names one")
    empty = Deployment(tokens={})
    for name, spec in sorted(specs.items()):
        try:
            render(spec, deployment=empty, allow_unmirrored=True)
            check(False, f"{name}: an EMPTY overlay must refuse")
        except SpecError as exc:
            check("answers none of" in str(exc),
                  f"{name}: an unanswered <<TOKEN>> is REFUSED, never rendered "
                  f"as itself or as nothing — either way the job would fail on "
                  f"the node, in a queue slot, with an error that reads like a "
                  f"staging mistake instead of a missing overlay")
        break                                   # one is the proof; nine is noise
    for tokens, why in (
            ({"lower_case": "x"},
             "a token name that is not UPPER_SNAKE is refused — nothing can "
             "reference it, so it resolves nothing"),
            ({"RAID": ""},
             "an EMPTY deployment value is refused: it renders a path with a "
             "hole in it and fails on the node rather than here"),
            ({"RAID": "<<RAID>>/x"},
             "a self-referential value is refused — substitution is ONE pass, "
             "deliberately, so an overlay cannot loop or resolve order-"
             "dependently")):
        try:
            Deployment(tokens=tokens)
            check(False, f"{why} — must be refused")
        except (ValidationError, ValueError):
            check(True, why)
    try:
        load_deployment(Path("/nonexistent/jobs-v21-deployment.json"))
        check(False, "a missing overlay file must be a named refusal")
    except SpecError as exc:
        check("gitignored overlay" in str(exc),
              "a MISSING overlay file names itself and says where the values "
              "belong, rather than raising an OSError nobody can act on")

    print("== selftest 9: the deployed scripts of record, byte for byte ==")
    fidelity = [n for n, s in specs.items() if s.lane in MIRRORED_LANES]
    if not fidelity:
        skip("NAMED SKIP — mirror sweep: the FIDELITY SET is absent, which is "
             "the normal clean-checkout state. The specs that reproduce the "
             "deployed scripts embed their verbatim prose and therefore live "
             "outside this repo (desk ruling 2026-08-01, rake M46: filenames "
             "AND content are the sanitization surface). Nothing here is "
             "unproved — it is UNRUN, and it runs at the desk with: python -m "
             "metabasis.jobs_v21 --selftest --spec-dir <STAGING>/jobs-v21-specs "
             "--mirror-dir <MIRROR_ROOT> --deployment <STAGING>/"
             "jobs-v21-deployment.json")
    elif mirror_root is not None and not real_deployment:
        skip("NAMED SKIP — mirror sweep: --mirror-dir was given but no real "
             "--deployment. Rendering under the selftest's NEUTRAL token "
             "values cannot reproduce the deployed bytes and a comparison "
             "against them would fail for a reason that is not a defect")
    elif mirror_root is None:
        skip("NAMED SKIP — mirror sweep: the fidelity set is loaded but no "
             "--mirror-dir was given. The pulled-back node scripts live under "
             "the gitignored staging tree; without them there is nothing to "
             "compare AGAINST, and a silent pass here would be the worst "
             "possible outcome. Run: python -m metabasis.jobs_v21 --selftest "
             "--spec-dir <STAGING>/jobs-v21-specs --mirror-dir <MIRROR_ROOT>")
    else:
        rels = mirror_relpaths(root)
        swept = 0
        for name, spec in sorted(specs.items()):
            if spec.lane not in MIRRORED_LANES:
                continue
            if name not in rels:
                skip(f"NAMED SKIP — {name}: no mirror_relpath sidecar, so this "
                     f"spec names no deployed ancestor to be compared against")
                continue
            result = compare(spec, Path(mirror_root), deployment=dep,
                             mirror_relpath=rels[name])
            swept += 1
            check(result.identical,
                  f"{name}: rendered == deployed {rels[name]} "
                  f"({result.mirror_sha256[:12]}…)"
                  + ("" if result.identical
                     else " — " + result.differences[0].detail))
            if not result.identical and result.differences[0].diff:
                for line in result.differences[0].diff[:40]:
                    print(f"        {line}")
        check(swept > 0, f"{swept} deployed script(s) of record swept")

    print(f"\nselftest: TOTAL — checks run: {checks} "
          f"({len(skips)} named skip(s)), {len(failures)} failure(s)")
    for line in skips:
        print(f"  SKIPPED: {line.splitlines()[0]}")
    for line in failures:
        print(f"  FAILED: {line}")
    return 1 if failures else 0


# ---------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only verification of the renderer, the spec, the "
                         "conversion and the comparison harness; no torch")
    ap.add_argument("--mirror-dir", type=Path, default=None,
                    help="the pulled-back node job tree under the gitignored "
                         "staging mirror. With --selftest it turns the "
                         "named-skipped mirror sweep into a real one")
    ap.add_argument("--deployment", type=Path, default=DEPLOYMENT_DEFAULT,
                    help="the gitignored overlay that answers the specs' "
                         "<<TOKEN>> placeholders (project root, venv, weights "
                         "store, RAID, node). Defaults beside the node mirror "
                         "under staging/; its absence is a named refusal")
    ap.add_argument("--spec-dir", type=Path, default=SPEC_ROOT,
                    help="where the spec fixtures live. The tracked default "
                         "holds only the SYNTHETIC examples; the FIDELITY set "
                         "(the specs that reproduce the deployed scripts, prose "
                         "and all) lives in the gitignored "
                         "staging/jobs-v21-specs/ per the 2026-08-01 desk "
                         "ruling. Point this there to run the byte-comparison")
    ap.add_argument("--list", action="store_true",
                    help="the shipped specs, their lanes and their mirrors")
    ap.add_argument("--render", metavar="SPEC", default=None,
                    help="render one shipped spec (by filename stem) to stdout, "
                         "or to --out-dir")
    ap.add_argument("--spec-file", type=Path, default=None,
                    help="render a spec from a file instead of the shipped set")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write the rendered script here instead of stdout")
    ap.add_argument("--memory-check", choices=("inline-flat", "capacity-module",
                                               "none"), default=None,
                    help="override the spec's memory mode. 'inline-flat' is "
                         "FIDELITY (what the deployed script carries); "
                         "'capacity-module' is the converted form")
    ap.add_argument("--fidelity", action="store_true",
                    help="render the DEPLOYED ancestor rather than a new-style "
                         "job: the ancestor's own OMP_NUM_THREADS and no "
                         "ruling note. Without it a render carries the ruled "
                         f"default of record (OMP_NUM_THREADS="
                         f"{RULED_OMP_NUM_THREADS}, Luxia 2026-08-01). "
                         "--compare-mirror always renders in fidelity mode; "
                         "this flag is for reproducing a deployed script by hand")
    ap.add_argument("--allow-unmirrored", action="store_true",
                    help="render a lane with no deployed script of record. FOR "
                         "REVIEW ONLY — the output says so in its own header")
    ap.add_argument("--conversion-diff", metavar="SPEC", default=None,
                    help="the inline-×1.6 -> metabasis.capacity diff for one "
                         "per-model spec")
    ap.add_argument("--fire-line", metavar="SPEC", default=None,
                    help="PRINT the `heimdall submit` line for one shipped "
                         "spec. Nothing is submitted: this module has no "
                         "network code and no subprocess import, and the CLI is "
                         "the submission interface of record")
    ap.add_argument("--node", default=None,
                    help="--fire-line: the node to pin. NO DEFAULT — a node "
                         "name is a deployment fact and a wrong one is a job "
                         "that runs against the wrong trees (for a GPU job "
                         "--node is a hard constraint; for a 0-GPU job it is "
                         "only a preference, guide §4.7)")
    ap.add_argument("--gpus", type=int, default=None,
                    help="--fire-line: GPU count. Defaults to the lane's shape "
                         "— the fan-out for a sharded vector build, 1 for a "
                         "single-card one, 0 for a fit")
    ap.add_argument("--jobs-dir", default=None,
                    help="--fire-line: the node-side directory the rendered "
                         "script is deployed into. NO DEFAULT, for the same "
                         "reason NodePaths has none")
    ap.add_argument("--after", action="append", default=[], metavar="JOB_ID",
                    help="--fire-line: one flag PER dependency (never "
                         "comma-joined — Heimdall would store the literal "
                         "string as one unknown id and queue forever)")
    ap.add_argument("--compare-mirror", type=Path, default=None, metavar="ROOT",
                    help="byte-compare every mirrored spec against a "
                         "pulled-back node job tree; nonzero exit on any "
                         "mismatch. Needs --spec-dir <STAGING>/jobs-v21-specs "
                         "and --deployment: the tracked spec dir holds only "
                         "synthetic examples, which have no deployed ancestor")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output for --compare-mirror/--list")
    args = ap.parse_args(argv)

    if args.selftest:
        #  RAKE M45: a sweep must catch SystemExit and must always reach its
        #  terminal TOTAL line. A `sys.exit()` raised out of any checked code
        #  path would otherwise truncate the log at whatever check happened to
        #  be running, and a truncated sweep log reads exactly like a short one.
        try:
            #  The overlay is OPTIONAL here and required nowhere else: without
            #  it the selftest runs under neutral values and NAMES the mirror
            #  sweep as skipped, which is what lets a clean checkout with no
            #  cluster attached still verify everything it honestly can.
            try:
                real = load_deployment(args.deployment)
            except SpecError:
                real = None
            return selftest(args.mirror_dir, real, args.spec_dir)
        except SystemExit as exc:
            print(f"\nselftest: TOTAL — ABORTED, a checked code path called "
                  f"sys.exit({exc.code}) and unwound the sweep. Everything "
                  f"after that point is UNRUN, not passed")
            return 1
        except Exception as exc:                        # noqa: BLE001
            print(f"\nselftest: TOTAL — ABORTED by "
                  f"{type(exc).__name__}: {exc}. Everything after that point "
                  f"is UNRUN, not passed")
            return 1

    try:
        specs = load_specs(args.spec_dir)
    except SpecError as exc:
        print(f"SPEC-MALFORMED: {exc}", file=sys.stderr)
        return 2

    deployment: Optional[Deployment] = None
    if not (args.list or args.fire_line):
        try:
            deployment = load_deployment(args.deployment)
        except SpecError as exc:
            print(f"DEPLOYMENT-MISSING: {exc}", file=sys.stderr)
            return 2

    if args.list:
        rels = mirror_relpaths(args.spec_dir)
        rows = [{"spec": n, "lane": s.lane, "renders": s.name,
                 "mirrored": s.lane in MIRRORED_LANES,
                 "mirror_relpath": rels.get(n, "")}
                for n, s in sorted(specs.items())]
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            width = max((len(r["spec"]) for r in rows), default=4)
            for row in rows:
                flag = "mirrored" if row["mirrored"] else "UNMIRRORED"
                print(f"{row['spec']:<{width}}  {row['lane']:<16} {flag:<10} "
                      f"{row['renders']}")
        return 0

    if args.compare_mirror is not None:
        rels = mirror_relpaths(args.spec_dir)
        results: list[Comparison] = []
        errored: list[str] = []
        for name, spec in sorted(specs.items()):
            if spec.lane not in MIRRORED_LANES:
                continue
            #  RAKE M45: one spec that raises (or calls sys.exit) must not eat
            #  the rest of the sweep or its TOTAL line — an interrupted sweep
            #  log is indistinguishable from a short one.
            try:
                assert deployment is not None
                results.append(compare(spec, args.compare_mirror,
                                       deployment=deployment,
                                       mirror_relpath=rels.get(name)))
            except SystemExit as exc:
                errored.append(f"{name}: called sys.exit({exc.code}) mid-sweep")
            except Exception as exc:                    # noqa: BLE001
                errored.append(f"{name}: {type(exc).__name__}: {exc}")
        if args.json:
            print(json.dumps({"comparisons": [r.model_dump() for r in results],
                              "errored": errored}, indent=1))
        else:
            for result in results:
                verdict = "IDENTICAL" if result.identical else "DIFFERS"
                print(f"{verdict:<10} {result.spec_name}  "
                      f"rendered {result.rendered_sha256[:12]}…  "
                      f"mirror {result.mirror_sha256[:12] or '(absent)'}…")
                for difference in result.differences:
                    print(f"           {difference.detail}")
                    for line in difference.diff:
                        print(f"           {line}")
            for line in errored:
                print(f"ERRORED    {line}")
        bad = [r for r in results if not r.identical]
        print(f"\nTOTAL: {len(results)} compared, {len(bad)} DIFFER, "
              f"{len(errored)} errored")
        if not results and not errored:
            #  A sweep that compared NOTHING must not exit 0. The tracked spec
            #  dir holds only synthetic examples, which have no deployed
            #  ancestor — so this is the "you pointed me at the wrong specs"
            #  case, and it reads exactly like a clean sweep unless it is said.
            print(f"COMPARE-REFUSED: no spec in {args.spec_dir} belongs to a "
                  f"mirrored lane, so nothing was compared. The fidelity set "
                  f"lives in the gitignored staging/jobs-v21-specs/ (desk "
                  f"ruling 2026-08-01) — pass --spec-dir", file=sys.stderr)
            return 2
        return 3 if (bad or errored) else 0

    if args.fire_line:
        spec = specs.get(args.fire_line)
        if spec is None:
            print(f"SPEC-MALFORMED: no shipped spec named {args.fire_line!r} "
                  f"(have: {', '.join(sorted(specs))})", file=sys.stderr)
            return 2
        missing = [f"--{n.replace('_', '-')}" for n in ("node", "jobs_dir")
                   if getattr(args, n) is None]
        if missing:
            print(f"SUBMISSION-REFUSED: {', '.join(missing)} must be given — "
                  f"neither has a default, because a wrong node or a wrong "
                  f"deploy directory is a job that runs against the wrong "
                  f"trees", file=sys.stderr)
            return 2
        paths = spec.paths
        default_gpus = (0 if spec.lane == "fits"
                        else getattr(getattr(spec, "memory", None), "n_cards", 1)
                        if spec.lane == "vector-model"
                        else len([d for d in getattr(spec, "shard_across", "")
                                  .split(",") if d.strip()]) or 1)
        try:
            submission = SubmissionSpec(
                job_script=f"{args.jobs_dir}/{spec.name}",
                name=f"mb-{Path(spec.name).stem}"[:60],
                gpus=args.gpus if args.gpus is not None else int(default_gpus),
                node=args.node,
                workdir=f"{paths.root}/{paths.code_dirname}",
                after=list(args.after))
        except (ValidationError, ValueError) as exc:
            print(f"SUBMISSION-REFUSED: {exc}", file=sys.stderr)
            return 2
        print("# NOT SUBMITTED — read this, then fire it yourself. The CLI is "
              "the submission")
        print("# interface of record; the coordinator HTTP API is READ-ONLY "
              "verification.")
        sys.stdout.write(submission.render())
        return 0

    if args.conversion_diff:
        spec = specs.get(args.conversion_diff)
        if spec is None:
            print(f"SPEC-MALFORMED: no shipped spec named "
                  f"{args.conversion_diff!r} (have: {', '.join(sorted(specs))})",
                  file=sys.stderr)
            return 2
        if not isinstance(spec, VectorModelSpec):
            print(f"SPEC-MALFORMED: {args.conversion_diff} is lane "
                  f"{spec.lane!r}; only the per-model vector lane carries the "
                  f"inline heuristic this converts", file=sys.stderr)
            return 2
        assert deployment is not None
        for line in conversion_diff(spec, deployment):
            print(line)
        return 0

    if args.render or args.spec_file:
        try:
            spec = (load_spec(args.spec_file) if args.spec_file
                    else specs[args.render])
        except (SpecError, KeyError) as exc:
            print(f"SPEC-MALFORMED: {exc}", file=sys.stderr)
            return 2
        try:
            assert deployment is not None
            text = render(spec, deployment=deployment,
                          memory_check=args.memory_check,
                          allow_unmirrored=args.allow_unmirrored,
                          fidelity=args.fidelity)
        except (TemplateError, SpecError) as exc:
            print(f"RENDER-REFUSED: {exc}", file=sys.stderr)
            return 2
        if args.out_dir:
            out = Path(args.out_dir)
            try:
                out.mkdir(parents=True, exist_ok=True)
                (out / spec.name).write_text(text, encoding="utf-8")
            except OSError as exc:
                print(f"RENDER-REFUSED: cannot write into {out} ({exc})",
                      file=sys.stderr)
                return 2
            print(f"wrote {out / spec.name}  sha256 {sha256_text(text)}")
        else:
            sys.stdout.write(text)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
