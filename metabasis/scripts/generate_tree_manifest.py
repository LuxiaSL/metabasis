"""ONE canonical sha256 manifest generator — the code the per-store shell
incantations retire into (optimization-pass item 8).

Every manifest in this campaign has been produced by a hand-typed pipeline of the
shape `find <tree> -type f -print0 | sort -z | xargs -0 sha256sum > <manifest>`,
re-typed per store, per session, per node. Four rakes were filed against that
habit inside three days — M38, M42, M43 and M46's filename rider — and every one
of them is a property of the GENERATOR, not of the data. A rake that can only be
paid by remembering to type a flag gets paid until the session someone forgets.
This module is where those flags stop being remembered and start being enforced.

THE FOUR RULES, AS CODE
-----------------------
  * **M38 — symlink blindness.** `find <tree> -type f` never descends a symlinked
    directory and never lists a symlinked file, so under the /models placement
    rule (where every relocated tree is reached through a symlink) a naive
    regeneration silently DROPS the relocated store and the diff reads as
    deletion. The walk here is `os.walk(followlinks=True)` — `find -L` semantics —
    with a (st_dev, st_ino) visit guard so a link cycle terminates instead of
    spinning (M38(a)). `compare_to_previous` implements M38(b): a manifest that
    SHRANK after a relocation is reported as this rake first and a deletion
    second.
  * **M42 — self-listing.** A file cannot contain its own hash, so a manifest
    that lists itself can never verify clean and every future checker must
    adjudicate a guaranteed false alarm. The output path is excluded from its own
    enumeration unconditionally (not by a glob that a renamed output could slip
    past — by identity, `os.path.samefile`-grade comparison of resolved paths),
    the written artifact is re-parsed after the write and a self-row there is a
    HALT, and `verify_manifest` classifies a self-row as a GENERATOR defect with
    its own exit code rather than as a digest failure (M42(c): the artifacts are
    innocent).
  * **M43 — scratch inside the tree.** A `MANIFEST-….sha256.tmp` staged INSIDE the
    tree being hashed self-includes as an empty-file row, and the M42 exclusion
    misses it because the suffix differs. Two defences: nothing at all is written
    during the walk (hashing completes in memory first, so there is no scratch
    file to find), and the atomic-rename scratch is placed in a directory proven
    to be OUTSIDE the hashed tree. A tree that ALREADY contains scratch files is
    a refusal (`ScratchInTreeError`) until they are named in `exclude_globs` —
    "explicitly excluded", never silently swept. `expect_rows` implements
    M43(b), the row-count-versus-derivation check that caught the original;
    `EMPTY_FILE_SHA256` rows are flagged on sight per M43(c).
  * **M46 — sanitization, filenames included.** `grep -c` exits 0 when it FINDS
    matches, so a sweep riding inside a `&&` chain publishes exactly when it
    fails; the rider that got filed sent a manifest row whose FILENAME carried an
    infrastructure alias to a remote. `scan_paths` is a zero-assert
    (`! grep -qE …`): any hit is a nonzero exit and the manifest is NOT WRITTEN.
    Filenames are content, so PATHS are scanned as well as bytes. `SANITIZE_MODE
    = "exclude"` (the desk's redaction lane) REFUSES to run without an
    `unsanitized_copy` destination, because M46(c) requires the full copy to stay
    desk-side for coverage to remain honest.

NO PROJECT-SPECIFIC PATTERNS LIVE IN THIS FILE, deliberately. A regex that
matches an infrastructure alias contains that alias, and this file is committed
to a public remote — shipping the sweep's pattern list would leak precisely the
strings the sweep exists to keep off the remote. `GENERIC_PATH_PATTERNS` holds
only patterns that are generic by construction (home-directory paths, e-mail
addresses, well-known credential prefixes); the campaign's own patterns are
loaded at run time from a gitignored desk-side file via `--patterns-file`.

BYTE COMPATIBILITY is the compatibility contract: output is exactly what GNU
`sha256sum` writes — `<64 hex><two spaces><path>\\n`, with GNU's leading-backslash
escape for the pathological names — so `sha256sum --check` remains the checker of
record and this module's `--verify` is a convenience, not a new format. `#`
header lines are supported because `manifests/collection.sha256` carries three of
them and GNU `sha256sum --check` (coreutils 9.5, measured) silently ignores them.

ORDERING is byte-wise (`LC_ALL=C sort`) by default, because that is the only
ordering that reproduces on another machine. NOTE FOR THE DESK, flagged not
fixed: `manifests/outputs.sha256` is NOT byte-sorted — it was produced by a
`sort -z` under a UTF-8 locale, whose collation ignores punctuation (the first
divergence is `…/a5_vectors_3b_14k/…` sorting before `…/a5_vectors_3b/…`). A
regeneration with the default ordering therefore reorders rows without changing
one digest, which is a large but content-free diff at the next freeze point.
`sort="locale"` reproduces the historical order for a minimal-diff regeneration;
it is offered, labelled non-portable, and is nobody's default.

CPU self-test (no weights, no GPU, no data tree, no torch):

    python -m metabasis.scripts.generate_tree_manifest --selftest

RAKE M44 — THE CONFIGURATION MATRIX IS THE MERGE BAR. Every reported count names
the configuration it was measured in. This selftest is data-tree-independent by
construction (every fixture is built in a `TemporaryDirectory`, nothing resolves
a banked artifact relative to cwd — the dcbe7d7 pattern), so the {data tree, no
data tree} axis is vacuous here and is asserted vacuous rather than assumed. The
axes that are NOT vacuous are {`sha256sum` on PATH, absent} — the byte-
compatibility blocks cross-check against the real binary — and {`read_composed_
predictions` importable, not} for the parser-agreement block. Each degrades to a
NAMED skip carrying its triggering condition, counted in the tail. Per M45 this
module's `selftest()` never calls `sys.exit`: it RETURNS a code and raises
`ManifestGeneratorError` on internal HALTs, so an all-module sweep survives it.

Usage:

    # generate (the shared lane; `--dry-run` prints the plan and writes nothing)
    python -m metabasis.scripts.generate_tree_manifest \\
        --tree outputs --output manifests/outputs.sha256 \\
        --patterns-file /desk/side/sanitization-patterns.txt \\
        --compare-previous manifests/outputs.sha256 --expect-rows 16008

    # verify (in-process `sha256sum --check` semantics, with M42 diagnosis)
    python -m metabasis.scripts.generate_tree_manifest \\
        --verify manifests/outputs.sha256 --anchor .

    # sanitization sweep alone, as its own BLOCKING command (M46(a))
    python -m metabasis.scripts.generate_tree_manifest \\
        --scan-only manifests/outputs.sha256 --patterns-file …

Exit codes: 0 clean · 1 a HALT or a digest failure · 2 usage (argparse) ·
3 the DATA verified clean but the MANIFEST is defective (a self-row: M42) ·
4 a sanitization hit (the zero-assert fired).
"""
from __future__ import annotations

import argparse
import errno
import fnmatch
import hashlib
import locale
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_tree_manifest")

# ------------------------------------------------------------------ constants
#: GNU `sha256sum` writes two spaces between digest and path in text mode. This
#: is the byte-compatibility contract; a single space is a different format and
#: `sha256sum --check` reads the second space as part of the filename.
ROW_SEPARATOR = "  "
#: sha256 of the empty string. Rake M43(c): its appearance in any manifest is a
#: red flag worth a scan on sight — it is what a scratch file staged inside the
#: hashed tree hashes to when `find` creates-then-hashes it.
EMPTY_FILE_SHA256 = ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b"
                     "7852b855")
#: The name prefix a manifest generator must exclude from its own enumeration
#: (rake M42(a)). Mirrors `read_composed_predictions.MANIFEST_NAME_PREFIX`; the
#: selftest asserts the two agree whenever that module is importable, so the
#: convention cannot drift into two values.
MANIFEST_NAME_PREFIX = "MANIFEST-"
#: Excluded by default, matching banked practice: `manifests/outputs.sha256`
#: contains ZERO rows matching either glob (measured, 16008 rows), so these
#: defaults reproduce the existing manifests rather than changing them.
DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = (f"{MANIFEST_NAME_PREFIX}*", "*.sha256")
#: A tree containing any of these is REFUSED (rake M43) until they are named in
#: `exclude_globs`. Scratch is a fact about the tree, not about the generator:
#: hashing a half-written file produces a digest of record for a file that never
#: existed in that state.
SCRATCH_GLOBS: tuple[str, ...] = (
    "*.tmp", "*.temp", "*.partial", "*.part", "*.crdownload", "*.swp", "*.swo",
    "*~", ".*.sw?", ".nfs*", "*.lock", "#*#", "*.rsync-partial")
#: Names whose bytes cannot be hashed reproducibly / whose escaping is a hazard.
#: Refused by default; `escape_odd_names=True` opts into GNU's escape for the two
#: characters GNU actually escapes (backslash, newline). Control characters stay
#: refused in both modes — nothing in this campaign's trees has ever carried one.
_ODD_NAME_CHARS = "\\\n"
#: every C0 control except newline (which the escape path handles), plus DEL.
_CONTROL_NAME_RE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
#: Read granularity for hashing. 1 MiB matches `read_composed_predictions.sha256_of`.
_HASH_CHUNK = 1 << 20
#: Bytes-scanning cap. A sanitization sweep over a 30 MB state bank is a waste of
#: an hour; over a JSON provenance sidecar it is the point. Files above the cap
#: are reported as NOT SCANNED rather than silently skipped (never a quiet pass).
DEFAULT_BYTES_SCAN_MAX = 4 << 20

#: Generic-by-construction patterns only — see the module docstring on why no
#: campaign-specific alias may ever be written here. Names are stable so a report
#: can say WHICH pattern fired without quoting the match.
#:
#: OPT-IN (`--generic-patterns`), never automatic, and tuned against the wolf:
#: a marker set that fires on honest artifacts gets switched off within a week,
#: and a switched-off sweep is worse than none. Two over-matches were found by
#: this module's own CLI smoke and are fixed here rather than documented as
#: quirks: a relative row `store/home/notes.txt` is NOT an absolute home path
#: (the `/home` must begin the string or follow a non-filename character), and
#: `bank@v1.2.json` is NOT an e-mail address (common file extensions are
#: excluded from the TLD).
GENERIC_PATH_PATTERNS: dict[str, str] = {
    "home-directory-path": r"(?:^|(?<=[^A-Za-z0-9_-]))/home/[^/\s]+",
    "macos-home-path": r"(?:^|(?<=[^A-Za-z0-9_-]))/Users/[^/\s]+",
    "email-address": (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\."
                      r"(?!json\b|jsonl\b|npz\b|npy\b|txt\b|md\b|py\b|csv\b"
                      r"|tsv\b|log\b|yaml\b|yml\b|toml\b|sha256\b|png\b|pdf\b"
                      r"|gz\b|zip\b|parquet\b|pt\b|safetensors\b)[A-Za-z]{2,}"),
    "anthropic-api-key": r"sk-ant-[A-Za-z0-9_\-]{8,}",
    "openai-api-key": r"\bsk-[A-Za-z0-9]{20,}",
    "aws-access-key-id": r"\bAKIA[0-9A-Z]{16}\b",
    "private-key-block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}
#: The FROZEN roster of generic pattern classes. The selftest asserts
#: `GENERIC_PATH_PATTERNS` has exactly these keys, so adding a campaign-specific
#: pattern to this file fails a check instead of quietly shipping the alias it
#: is written from. Campaign patterns load from `--patterns-file` — always.
GENERIC_PATTERN_NAMES: tuple[str, ...] = (
    "home-directory-path", "macos-home-path", "email-address",
    "anthropic-api-key", "openai-api-key", "aws-access-key-id",
    "private-key-block")


# -------------------------------------------------------------------- errors
class ManifestGeneratorError(Exception):
    """Base HALT. Raised, never `sys.exit`-ed (rake M45): a module selftest that
    exits kills an all-module sweep silently, because SystemExit is not
    Exception and every later module never runs."""


class ScratchInTreeError(ManifestGeneratorError):
    """Rake M43: the tree to be hashed contains scratch/temporary files."""


class SanitizationLeakError(ManifestGeneratorError):
    """Rake M46: a sanitization pattern matched a path or the bytes of a covered
    artifact. The zero-assert fired; nothing is written."""


class RowCountMismatchError(ManifestGeneratorError):
    """Rake M43(b): the row count disagrees with the caller's derivation. This is
    the check that caught the 2919-vs-2918 scratch row before it was promoted."""


class UnreadableArtifactError(ManifestGeneratorError):
    """A covered file could not be read. A manifest that silently omits it is the
    entire M38 failure mode wearing a different hat, so this HALTs by default."""


class ManifestFormatError(ManifestGeneratorError):
    """A manifest line is not a `<digest>  <path>` row in either banked dialect."""


# -------------------------------------------------------------------- models
class SanitizationPattern(BaseModel):
    """One named regex for the M46 sweep. Named so a finding can be reported
    WITHOUT quoting the matched text — the string is exactly what must not
    propagate, and a refusal message is a place strings propagate to."""
    model_config = ConfigDict(frozen=True)

    name: str
    pattern: str
    #: scan row PATHS with this (M46(b): filenames are content)
    paths: bool = True
    #: scan covered artifacts' BYTES with this (costlier; opt-in per pattern)
    contents: bool = False

    @field_validator("pattern")
    @classmethod
    def _compilable(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"not a valid regex: {value!r} ({exc})") from exc
        return value

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern)


class SanitizationHit(BaseModel):
    """A match. `redacted` is what a log may carry; `matched` is populated only
    when the caller explicitly asked to see it (`--show-matches`, desk-side)."""
    pattern_name: str
    where: Literal["path", "bytes"]
    row_name: str
    line_number: Optional[int] = None
    redacted: str
    matched: Optional[str] = None

    def describe(self) -> str:
        return (f"[{self.where}] pattern {self.pattern_name!r} matched "
                f"{self.redacted}"
                + (f" (match: {self.matched!r})" if self.matched else ""))


class HashedFile(BaseModel):
    """One covered artifact: its manifest row name, its digest, and the identity
    facts a concurrency check needs."""
    #: the row name as it will be written — relative to the ANCHOR, POSIX slashes
    name: str
    digest: str
    size: int
    mtime_ns: int
    #: the on-disk path walked to (through symlinks, not resolved) — a report can
    #: quote it, and it is NOT what goes in the manifest
    walked_path: str
    #: True when the file or one of its parent directories was reached through a
    #: symlink: the rows `find -type f` would have silently dropped (rake M38)
    via_symlink: bool = False


class ManifestDiff(BaseModel):
    """Row-level comparison against the previous manifest of record."""
    n_previous: int
    n_current: int
    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []

    @property
    def delta(self) -> int:
        return self.n_current - self.n_previous

    @property
    def shrank(self) -> bool:
        return self.n_current < self.n_previous


class ManifestSpec(BaseModel):
    """The generation contract. Pydantic because every field here is a rule that
    has already been broken once by a shell flag someone forgot to type."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: the tree to hash. Rows are named RELATIVE TO `anchor`, so passing
    #: `tree=outputs, anchor=.` from the repo root reproduces `outputs/…` rows.
    tree: Path
    #: where the manifest is written. May sit inside `tree` (the per-staging
    #: `MANIFEST-*.sha256` case) — it is excluded from its own enumeration by
    #: identity either way (rake M42).
    output: Path
    #: the directory row names are relative to. Must be an ancestor of `tree`, so
    #: no row can carry a `../` escape. Defaults to the current directory, which
    #: is what the banked `find outputs -type f | xargs sha256sum` did.
    anchor: Path = Field(default_factory=Path.cwd)
    #: `#` header lines, written verbatim above the rows. GNU `sha256sum --check`
    #: ignores them (measured, coreutils 9.5); `manifests/collection.sha256`
    #: carries three.
    headers: list[str] = []
    exclude_globs: list[str] = list(DEFAULT_EXCLUDE_GLOBS)
    #: matched against the row name, so a whole subtree can be dropped
    exclude_path_globs: list[str] = []
    scratch_globs: list[str] = list(SCRATCH_GLOBS)
    sanitization_patterns: list[SanitizationPattern] = []
    #: "refuse" = the zero-assert HALTs and nothing is written (the default, and
    #: what M46(a) asks for). "exclude" = matching rows are dropped from the
    #: published manifest and the FULL manifest is written to `unsanitized_copy`,
    #: which is therefore mandatory in that mode (M46(c)).
    sanitize_mode: Literal["refuse", "exclude"] = "refuse"
    unsanitized_copy: Optional[Path] = None
    #: bytes-scanning is opt-in and capped; see `DEFAULT_BYTES_SCAN_MAX`
    scan_bytes: bool = False
    bytes_scan_max: int = DEFAULT_BYTES_SCAN_MAX
    sort: Literal["byte", "locale"] = "byte"
    #: rake M43(b): the caller's independent derivation of the row count. A
    #: mismatch is a HALT before anything is promoted to head.
    expect_rows: Optional[int] = None
    #: rake M38(b): the previous manifest, for the row-count-and-rows comparison
    compare_previous: Optional[Path] = None
    #: broken symlinks are what `find -L -type f` also omits; strict makes them a
    #: HALT instead of a reported finding
    strict_broken_symlinks: bool = False
    #: an unreadable covered file HALTs unless this is set, in which case it is
    #: reported LOUDLY and omitted
    skip_unreadable: bool = False
    #: a file whose (size, mtime) changed between hashing and the post-pass
    #: re-stat HALTs: its digest is of a state that is already gone
    allow_concurrent_writes: bool = False
    escape_odd_names: bool = False
    dry_run: bool = False

    @field_validator("tree", "output", "anchor", "unsanitized_copy")
    @classmethod
    def _expand(cls, value: Optional[Path]) -> Optional[Path]:
        return None if value is None else Path(os.path.expanduser(str(value)))

    @model_validator(mode="after")
    def _coherent(self) -> "ManifestSpec":
        if self.sanitize_mode == "exclude" and self.unsanitized_copy is None:
            raise ValueError(
                "sanitize_mode='exclude' requires unsanitized_copy: rake M46(c) — "
                "a full unsanitized copy of any redacted artifact stays desk-side "
                "(gitignored) so coverage remains honest. Excluding rows with no "
                "complete copy anywhere turns a redaction into data loss")
        if self.bytes_scan_max <= 0:
            raise ValueError("bytes_scan_max must be positive")
        if self.expect_rows is not None and self.expect_rows < 0:
            raise ValueError("expect_rows must be >= 0")
        return self


class ManifestBuildResult(BaseModel):
    """Everything the generation learned, whether or not it wrote anything."""
    manifest_path: str
    tree: str
    anchor: str
    rows: list[HashedFile] = []
    #: names excluded by glob or by the M42 self-identity rule, with the reason
    excluded: dict[str, str] = {}
    scratch_found: list[str] = []
    broken_symlinks: list[str] = []
    unreadable: list[str] = []
    changed_during_walk: list[str] = []
    empty_file_rows: list[str] = []
    symlinked_rows: list[str] = []
    sanitization_hits: list[SanitizationHit] = []
    bytes_not_scanned: list[str] = []
    diff: Optional[ManifestDiff] = None
    findings: list[str] = []
    written: bool = False
    unsanitized_copy_written: Optional[str] = None
    n_bytes_written: int = 0
    sort: str = "byte"

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def clean(self) -> bool:
        """No finding that a desk must adjudicate before promoting this to head."""
        return not (self.findings or self.sanitization_hits or self.unreadable
                    or self.changed_during_walk)

    def manifest_text(self, escape_odd_names: bool = False) -> str:
        return build_manifest_text([(r.digest, r.name) for r in self.rows],
                                   headers=[], escape_odd_names=escape_odd_names)


class ParsedRow(BaseModel):
    """One parsed manifest row, normalized across both banked dialects."""
    digest: str
    #: the path exactly as written, so a report can quote the manifest verbatim
    raw_name: str
    #: `raw_name` with `./`, a binary-mode `*` marker and GNU escaping removed —
    #: the key every lookup uses, because the two dialects name the same artifact
    name: str
    line_number: int
    is_self_row: bool = False


class ParsedManifest(BaseModel):
    """A parsed manifest with every M42 hazard surfaced as data rather than as a
    surprise inside a checker."""
    path: str
    dialect: Literal["bare", "dot-prefixed", "mixed", "empty"]
    rows: list[ParsedRow] = []
    self_rows: list[str] = []
    header_lines: list[str] = []
    unparsed_lines: list[str] = []

    @property
    def by_name(self) -> dict[str, str]:
        return {r.name: r.digest for r in self.rows if not r.is_self_row}

    @property
    def lists_itself(self) -> bool:
        return bool(self.self_rows)


class VerifyRow(BaseModel):
    name: str
    status: Literal["OK", "FAILED", "MISSING", "SELF-ROW"]
    expected: str
    computed: Optional[str] = None
    detail: str = ""


class VerifyResult(BaseModel):
    """`sha256sum --check` semantics in-process, plus the M42 diagnosis it lacks."""
    manifest_path: str
    anchor: str
    rows: list[VerifyRow] = []
    self_rows: list[str] = []
    unparsed_lines: list[str] = []
    findings: list[str] = []

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.rows if r.status == "OK")

    @property
    def failed(self) -> list[VerifyRow]:
        return [r for r in self.rows if r.status == "FAILED"]

    @property
    def missing(self) -> list[VerifyRow]:
        return [r for r in self.rows if r.status == "MISSING"]

    @property
    def data_clean(self) -> bool:
        """The ARTIFACTS verify. Deliberately independent of `manifest_clean`:
        rake M42(c) — a self-row is a generator defect and the data is innocent."""
        return not (self.failed or self.missing)

    @property
    def manifest_clean(self) -> bool:
        return not (self.self_rows or self.unparsed_lines)

    def exit_code(self) -> int:
        if not self.data_clean:
            return 1
        return 0 if self.manifest_clean else 3


# ----------------------------------------------------------------- the walk
def _visit_key(path: Path) -> Optional[tuple[int, int]]:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino)


def walk_tree_following_symlinks(
        root: Path) -> Iterator[tuple[Path, bool, Optional[str]]]:
    """`find -L <root> -type f` at Python grain (rake M38).

    Yields `(path, via_symlink, broken_reason)` for every regular file reachable
    from `root` WITH symlinks followed — both symlinked files and files inside
    symlinked directories, which are exactly the rows `find -type f` and
    `Path.rglob` silently drop. Under the /models placement rule the relocated
    tree is the normal case, so the naive walk's omission reads as a deletion.

    LINK CYCLES terminate (M38(a)): each directory's (st_dev, st_ino) is recorded
    and a second visit prunes that branch rather than raising. A cycle is a fact
    about someone else's tree, not a reason for a manifest run to die.

    BROKEN symlinks are yielded with a reason and no `path` guarantee, because
    `find -L -type f` also omits them — but omitting them SILENTLY is how a
    manifest quietly loses coverage, so they are reported.
    """
    if not root.exists():
        raise ManifestGeneratorError(
            f"tree does not exist: {root} — refusing to write a manifest of "
            f"nothing (an empty manifest at head is indistinguishable from a "
            f"deleted store)")
    if not root.is_dir():
        raise ManifestGeneratorError(f"tree is not a directory: {root}")

    seen: set[tuple[int, int]] = set()
    root_real = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        here = Path(dirpath)
        key = _visit_key(here)
        if key is None:                      # vanished or unreadable mid-walk
            dirnames[:] = []
            yield here, True, "unreadable directory"
            continue
        if key in seen:
            dirnames[:] = []                 # cycle or a second path to one tree
            continue
        seen.add(key)
        #: a directory is "reached through a symlink" if any component of the
        #: walked path differs from its resolved form
        dir_via_symlink = here.resolve() != _plain_resolve(here, root, root_real)
        dirnames.sort()
        for filename in sorted(filenames):
            path = here / filename
            via = dir_via_symlink or path.is_symlink()
            try:
                if not path.exists():        # dangling symlink: `find -L` omits
                    yield path, via, "broken symlink"
                    continue
                if not path.is_file():       # fifo, socket, device
                    yield path, via, "not a regular file"
                    continue
            except OSError as exc:           # pragma: no cover — permission race
                yield path, via, f"stat failed: {exc}"
                continue
            yield path, via, None


def _plain_resolve(path: Path, root: Path, root_real: Path) -> Path:
    """`path` with only the ROOT's own symlinks resolved — so a tree handed in by
    a symlinked name does not make every row read as symlinked."""
    try:
        rel = path.relative_to(root)
    except ValueError:                       # pragma: no cover — defensive
        return path.resolve()
    return root_real / rel


def sha256_of_file(path: Path) -> str:
    """Streamed sha256, 1 MiB at a time. Matches the banked helper exactly."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------- row formatting
def _escape_name(name: str) -> tuple[str, bool]:
    """GNU `sha256sum`'s escape: a line whose path contains a backslash or a
    newline is prefixed with `\\` and those two characters are escaped. Measured
    against coreutils 9.5, which writes `\\<digest>  back\\\\slash.txt`."""
    if not any(ch in name for ch in _ODD_NAME_CHARS):
        return name, False
    return name.replace("\\", "\\\\").replace("\n", "\\n"), True


def _unescape_name(name: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(name):
        char = name[index]
        if char == "\\" and index + 1 < len(name):
            nxt = name[index + 1]
            if nxt == "\\":
                out.append("\\")
                index += 2
                continue
            if nxt == "n":
                out.append("\n")
                index += 2
                continue
        out.append(char)
        index += 1
    return "".join(out)


def format_manifest_row(digest: str, name: str,
                        escape_odd_names: bool = False) -> str:
    """One byte-exact `sha256sum` row (no trailing newline).

    Refuses control characters unconditionally and refuses backslash/newline
    unless `escape_odd_names` — a path that needs escaping has never occurred in
    this campaign's trees, and a silently escaped row is a row whose name no
    downstream `grep` will match.
    """
    if not _SHA256_HEX_RE.match(digest):
        raise ManifestFormatError(
            f"not a lowercase 64-hex sha256: {digest!r} — rake M40: a digest is "
            f"either a complete verified value or an explicitly-labelled prefix, "
            f"and nothing in between goes in a manifest")
    if _CONTROL_NAME_RE.search(name):
        raise ManifestFormatError(
            f"path contains a control character: {name!r} — refused, because no "
            f"escaping convention here round-trips it through `sha256sum --check`")
    if name != name.lstrip() or not name:
        raise ManifestFormatError(
            f"path is empty or starts with whitespace: {name!r} — refused: the "
            f"row separator is whitespace, so such a name cannot be read back "
            f"out of its own manifest")
    escaped, needs_escape = _escape_name(name)
    if needs_escape and not escape_odd_names:
        raise ManifestFormatError(
            f"path contains a backslash or newline: {name!r} — refused by "
            f"default. Pass escape_odd_names=True to write GNU's escaped form "
            f"(`\\<digest>  <escaped path>`); nothing in this campaign's trees "
            f"has ever needed it, so the refusal is the finding")
    prefix = "\\" if needs_escape else ""
    return f"{prefix}{digest}{ROW_SEPARATOR}{escaped}"


def build_manifest_text(rows: Sequence[tuple[str, str]],
                        headers: Sequence[str] = (),
                        escape_odd_names: bool = False) -> str:
    """The full manifest body: `#` headers, then `<digest>  <path>` rows."""
    lines = [f"# {h}" if not h.startswith("#") else h for h in headers]
    lines.extend(format_manifest_row(digest, name, escape_odd_names)
                 for digest, name in rows)
    return "".join(f"{line}\n" for line in lines)


def _sort_key(sort: Literal["byte", "locale"]) -> Callable[[str], Any]:
    if sort == "byte":
        return lambda name: name.encode("utf-8", "surrogateescape")
    #: `sort -z` under a UTF-8 locale — reproduces `manifests/outputs.sha256`'s
    #: historical order, and is NOT portable across machines or locale versions.
    try:
        locale.setlocale(locale.LC_COLLATE, "")
    except locale.Error as exc:              # pragma: no cover — exotic env
        logger.warning("locale collation unavailable (%s); falling back to "
                       "byte order, which is the deterministic one anyway", exc)
        return lambda name: name.encode("utf-8", "surrogateescape")
    return locale.strxfrm


# ----------------------------------------------------------- the M46 sweep
def load_patterns_file(path: Path) -> list[SanitizationPattern]:
    """Read a desk-side sanitization pattern file. One pattern per line:

        # comment
        name = regex          # named
        regex                 # auto-named pattern-N
        contents: name = regex    # also scanned against file BYTES
        paths: name = regex       # paths only (the default)

    The file is deliberately NOT in this repository: a regex that matches an
    infrastructure alias contains that alias, and this module is committed to a
    public remote.
    """
    if not path.is_file():
        raise ManifestGeneratorError(
            f"sanitization pattern file absent: {path} — refusing to run a sweep "
            f"with no patterns, because an empty sweep reports a clean pass and "
            f"is the most convincing false negative available")
    patterns: list[SanitizationPattern] = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        scope: Literal["paths", "contents", "both"] = "paths"
        for marker, resolved in (("contents:", "contents"), ("paths:", "paths"),
                                 ("both:", "both")):
            if line.lower().startswith(marker):
                scope = resolved              # type: ignore[assignment]
                line = line[len(marker):].strip()
                break
        if "=" in line and not line.startswith("="):
            name, _, pattern = line.partition("=")
            name, pattern = name.strip(), pattern.strip()
        else:
            name, pattern = f"pattern-{number}", line
        if not pattern:
            raise ManifestGeneratorError(
                f"{path}:{number}: empty pattern — a blank regex matches "
                f"everything and would refuse every manifest")
        try:
            patterns.append(SanitizationPattern(
                name=name, pattern=pattern,
                paths=scope in ("paths", "both"),
                contents=scope in ("contents", "both")))
        except ValueError as exc:
            raise ManifestGeneratorError(f"{path}:{number}: {exc}") from exc
    if not patterns:
        raise ManifestGeneratorError(
            f"{path} defines no patterns — see above on empty sweeps")
    return patterns


def generic_patterns(contents: bool = False) -> list[SanitizationPattern]:
    """The generic-by-construction set. Never sufficient on its own: the
    campaign's own aliases live in a desk-side `--patterns-file`."""
    return [SanitizationPattern(name=name, pattern=pattern, paths=True,
                                contents=contents)
            for name, pattern in GENERIC_PATH_PATTERNS.items()]


def _redact(text: str, match: re.Match[str]) -> str:
    return f"{text[:match.start()]}⟨REDACTED⟩{text[match.end():]}"


def scan_paths(names: Sequence[str], patterns: Sequence[SanitizationPattern],
               show_matches: bool = False) -> list[SanitizationHit]:
    """Rake M46(b): FILENAMES ARE CONTENT. Zero-assert — the caller treats a
    non-empty return as `! grep -qE` firing, i.e. a refusal, never a count to
    print inside a `&&` chain."""
    hits: list[SanitizationHit] = []
    active = [(p, p.compiled()) for p in patterns if p.paths]
    for name in names:
        for pattern, compiled in active:
            match = compiled.search(name)
            if match is not None:
                hits.append(SanitizationHit(
                    pattern_name=pattern.name, where="path", row_name=name,
                    redacted=_redact(name, match),
                    matched=match.group(0) if show_matches else None))
    return hits


def scan_bytes(rows: Sequence[HashedFile],
               patterns: Sequence[SanitizationPattern],
               max_size: int = DEFAULT_BYTES_SCAN_MAX,
               show_matches: bool = False
               ) -> tuple[list[SanitizationHit], list[str]]:
    """The bytes half of the sweep. Returns `(hits, not_scanned)` — an oversized
    or undecodable file is REPORTED as not scanned, never counted as clean.

    Reads through `HashedFile.walked_path`, the path the walk actually reached
    (symlinks followed, rake M38), so a relocated tree is scanned rather than
    reported as unreadable.
    """
    hits: list[SanitizationHit] = []
    not_scanned: list[str] = []
    active = [(p, p.compiled()) for p in patterns if p.contents]
    if not active:
        return hits, not_scanned
    for row in rows:
        if row.size > max_size:
            not_scanned.append(f"{row.name} ({row.size} B > {max_size} B cap)")
            continue
        try:
            text = Path(row.walked_path).read_text(errors="replace")
        except OSError as exc:
            not_scanned.append(f"{row.name} (unreadable: {exc})")
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for pattern, compiled in active:
                match = compiled.search(line)
                if match is not None:
                    hits.append(SanitizationHit(
                        pattern_name=pattern.name, where="bytes",
                        row_name=row.name, line_number=number,
                        redacted=_redact(line[:200], match)[:240],
                        matched=match.group(0) if show_matches else None))
    return hits, not_scanned


def assert_no_sanitization_hits(hits: Sequence[SanitizationHit]) -> None:
    """The blocking zero-assert (M46(a)). This is a COMMAND of its own, not a
    rider: `grep -c` exits 0 when it finds matches, so the counting form publishes
    exactly when the sweep fails."""
    if hits:
        raise SanitizationLeakError(
            f"sanitization sweep found {len(hits)} hit(s) — rake M46, blocking. "
            + "; ".join(hit.describe() for hit in hits[:8])
            + ("" if len(hits) <= 8 else f" … (+{len(hits) - 8} more)"))


# ------------------------------------------------------------- the generator
def _is_within(path: Path, tree: Path) -> bool:
    """True when `path` is `tree` or lives beneath it, symlinks resolved on both
    sides. Used for the M43 scratch-placement proof, where a false negative is
    the entire rake."""
    try:
        resolved, tree_resolved = path.resolve(), tree.resolve()
    except OSError:                          # pragma: no cover — defensive
        return False
    return resolved == tree_resolved or tree_resolved in resolved.parents


def scratch_staging_dir(tree: Path, output: Path) -> Path:
    """A directory PROVEN to be outside `tree` (rake M43(a)).

    Preference order: the output's own parent (same filesystem as the final
    path, so the rename is atomic) · the tree's parent · the system temp dir.
    The last is always outside the tree but may be on another filesystem, in
    which case the rename degrades to a copy and says so.
    """
    for candidate in (output.parent, tree.parent, Path(tempfile.gettempdir())):
        try:
            if candidate.is_dir() and not _is_within(candidate, tree):
                return candidate
        except OSError:                      # pragma: no cover — defensive
            continue
    raise ManifestGeneratorError(          # pragma: no cover — needs / as tree
        f"no staging directory outside {tree} could be found; rake M43 forbids "
        f"staging the scratch file inside the tree being hashed")


def _atomic_write(text: str, destination: Path, tree: Path) -> int:
    """Write via a scratch file staged OUTSIDE `tree`, then rename into place.

    Nothing at all is written during the walk — the whole manifest is built in
    memory first — so the M43 failure mode (a scratch file inside the tree being
    created-then-hashed as an empty row) cannot occur even if this staging
    directory choice were wrong.
    """
    payload = text.encode("utf-8")
    staging = scratch_staging_dir(tree, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=".manifest-staging-",
                                         suffix=".sha256", dir=str(staging))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        try:
            os.replace(temp_path, destination)
        except OSError as exc:
            if getattr(exc, "errno", None) != errno.EXDEV:
                raise
            logger.warning(
                "staging dir %s is on another filesystem than %s: the rename "
                "degrades to a copy (not atomic). The M43 rule — scratch OUTSIDE "
                "the hashed tree — is still honoured", staging, destination)
            shutil.copyfile(temp_path, destination)
            temp_path.unlink(missing_ok=True)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    written = destination.read_bytes()
    if written != payload:                   # pragma: no cover — torn write
        raise ManifestGeneratorError(
            f"{destination}: written bytes differ from the intended manifest "
            f"({len(written)} vs {len(payload)} B) — the write was torn; the "
            f"previous head must be restored before anything is promoted")
    return len(payload)


def parse_manifest(path: Path) -> ParsedManifest:
    """Parse a sha256 manifest, accepting BOTH banked dialects explicitly.

    Rake M42(d): bare (`<digest>  name`) and `./`-prefixed with `#` headers are
    both of record in this tree. A `#` line is a COMMENT, counted and never
    parsed as a row — a two-word comment becoming a phantom artifact is the exact
    failure that rule was filed for. Duplicate names HALT (rake M18): which
    digest is of record cannot be guessed, and a dict that collapses two rows
    reports a false pass at n−1.
    """
    if not path.is_file():
        raise ManifestGeneratorError(f"manifest absent: {path}")
    resolved = path.resolve()
    rows: list[ParsedRow] = []
    self_rows: list[str] = []
    headers: list[str] = []
    unparsed: list[str] = []
    seen: dict[str, int] = {}
    dotted = bare = 0
    for number, line in enumerate(
            path.read_text(errors="replace").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            headers.append(stripped)
            continue
        escaped_row = stripped.startswith("\\")
        body = stripped[1:] if escaped_row else stripped
        parts = body.split(None, 1)
        if len(parts) != 2 or not _SHA256_HEX_RE.match(parts[0]):
            unparsed.append(f"line {number}: {line!r}")
            continue
        digest, raw_name = parts[0], parts[1]
        name = _unescape_name(raw_name) if escaped_row else raw_name
        if name.startswith("*"):
            name = name[1:]
        if name.startswith("./"):
            name = name[2:]
            dotted += 1
        else:
            bare += 1
        if name in seen:
            raise ManifestFormatError(
                f"{path}: duplicate row for {name!r} (lines {seen[name]} and "
                f"{number}) — rake M18: which digest is of record cannot be "
                f"guessed")
        seen[name] = number
        #  A SELF-ROW is judged by BASENAME first, resolution second: a
        #  manifest's anchor is not always its parent (rake M15), so a
        #  resolution-only test misses the self-row in every manifest anchored
        #  elsewhere — and missing one is the whole point of the check.
        is_self = Path(name).name == path.name
        if is_self:
            try:
                is_self = ((path.parent / name).resolve() == resolved
                           or Path(name).parent in (Path(""), Path(".")))
            except OSError:                  # pragma: no cover — unreadable
                is_self = True
        if is_self:
            self_rows.append(raw_name)
        rows.append(ParsedRow(digest=digest, raw_name=raw_name, name=name,
                              line_number=number, is_self_row=is_self))
    dialect: Literal["bare", "dot-prefixed", "mixed", "empty"] = (
        "empty" if not rows else
        "mixed" if dotted and bare else
        "dot-prefixed" if dotted else "bare")
    return ParsedManifest(path=str(path), dialect=dialect, rows=rows,
                          self_rows=self_rows, header_lines=headers,
                          unparsed_lines=unparsed)


def compare_to_previous(previous: Path,
                        rows: Sequence[HashedFile]) -> ManifestDiff:
    """Row-level diff against the previous manifest of record (rake M38(b))."""
    parsed = parse_manifest(previous)
    old = parsed.by_name
    new = {row.name: row.digest for row in rows}
    return ManifestDiff(
        n_previous=len(old), n_current=len(new),
        added=sorted(set(new) - set(old)),
        removed=sorted(set(old) - set(new)),
        changed=sorted(name for name in set(old) & set(new)
                       if old[name] != new[name]))


def generate_tree_manifest(spec: ManifestSpec) -> ManifestBuildResult:
    """THE shared generator. One function, every rule, no flags to remember.

    Order of operations is itself load-bearing:
      1. walk with `find -L` semantics (M38) and classify what is found;
      2. REFUSE on scratch in the tree (M43) before hashing a byte;
      3. hash everything IN MEMORY — nothing is written during the walk, so no
         scratch file can self-include (M43 again, structurally);
      4. re-stat and HALT on anything that changed under us;
      5. run the sanitization zero-assert over PATHS (and bytes on request) —
         M46, blocking, before any write;
      6. check the row count against the caller's derivation (M43(b));
      7. write LAST, atomically, from a staging dir proven outside the tree;
      8. re-parse the written artifact and HALT on a self-row (M42).
    """
    tree, anchor, output = spec.tree, spec.anchor, spec.output
    if not _is_within(tree, anchor) and tree.resolve() != anchor.resolve():
        raise ManifestGeneratorError(
            f"anchor {anchor} is not an ancestor of tree {tree}: every row would "
            f"carry a `../` escape, which no `sha256sum --check` run from the "
            f"anchor can resolve and no reader can interpret. Pass an anchor the "
            f"tree lives under (the banked convention is the repo root)")

    result = ManifestBuildResult(manifest_path=str(output), tree=str(tree),
                                 anchor=str(anchor), sort=spec.sort)
    output_resolved: Optional[Path]
    try:
        output_resolved = output.resolve()
    except OSError:                          # pragma: no cover — defensive
        output_resolved = None

    #  "Explicitly excluded" (rake M43) means a glob THIS CALLER supplied — a
    #  shipped default is not an acknowledgement of anything.
    explicit_exclusions = [g for g in spec.exclude_globs
                           if g not in DEFAULT_EXCLUDE_GLOBS]
    explicit_exclusions.extend(spec.exclude_path_globs)

    candidates: list[HashedFile] = []
    for path, via_symlink, broken in walk_tree_following_symlinks(tree):
        rel = os.path.relpath(path, anchor)
        name = Path(rel).as_posix()
        if broken is not None:
            if broken == "broken symlink":
                result.broken_symlinks.append(name)
            else:
                result.excluded[name] = broken
            continue
        #  M42 BY IDENTITY, not by glob: the output is excluded because it IS the
        #  output, so a manifest named anything at all still cannot list itself.
        if output_resolved is not None:
            try:
                if path.resolve() == output_resolved:
                    result.excluded[name] = "the manifest itself (rake M42)"
                    continue
            except OSError:                  # pragma: no cover — defensive
                pass
        basename = Path(name).name
        scratch = next((g for g in spec.scratch_globs
                        if fnmatch.fnmatch(basename, g)), None)
        explicit = next((g for g in explicit_exclusions
                         if fnmatch.fnmatch(basename, g)
                         or fnmatch.fnmatch(name, g)), None)
        if scratch is not None and explicit is None:
            #  SCRATCH IS CHECKED BEFORE THE EXCLUSION GLOBS, and this ordering
            #  is the rake itself. `MANIFEST-….sha256.tmp` matches the DEFAULT
            #  `MANIFEST-*` exclusion, so an exclusion-first generator drops it
            #  quietly and reports a clean run over a tree that is mid-write —
            #  M43's failure with the sign flipped, found by this module's own
            #  selftest. A default can therefore never swallow scratch; only a
            #  glob the CALLER added can, which is what "explicitly excluded"
            #  has to mean.
            result.scratch_found.append(f"{name} (matches {scratch!r})")
            continue
        excluded_by = explicit
        if excluded_by is None:
            excluded_by = next((g for g in spec.exclude_globs
                                if fnmatch.fnmatch(basename, g)), None)
        if excluded_by is not None:
            reason = f"excluded by glob {excluded_by!r}"
            if scratch is not None:
                reason += f" (scratch matching {scratch!r}, EXPLICITLY excluded)"
            result.excluded[name] = reason
            continue
        try:
            stat = path.stat()
            digest = sha256_of_file(path)
        except OSError as exc:
            result.unreadable.append(f"{name}: {exc}")
            continue
        candidates.append(HashedFile(name=name, digest=digest, size=stat.st_size,
                                     mtime_ns=stat.st_mtime_ns,
                                     walked_path=str(path),
                                     via_symlink=via_symlink))
        if via_symlink:
            result.symlinked_rows.append(name)
        if digest == EMPTY_FILE_SHA256:
            result.empty_file_rows.append(name)

    if result.scratch_found and spec.dry_run:
        result.findings.append(
            f"SCRATCH IN TREE (rake M43): {len(result.scratch_found)} file(s) — "
            f"a real run REFUSES on these; they are reported here because a dry "
            f"run exists to show the operator what a real run would hit: "
            + "; ".join(result.scratch_found[:8]))
    if result.scratch_found and not spec.dry_run:
        raise ScratchInTreeError(
            f"{len(result.scratch_found)} scratch/temporary file(s) inside "
            f"{tree} — rake M43. A manifest generated over a tree that contains "
            f"scratch hashes a half-written state as a digest of record, and the "
            f"M42 exclusion misses it whenever the suffix differs. Remove them, "
            f"or name them in exclude_globs EXPLICITLY: "
            + "; ".join(result.scratch_found[:8])
            + ("" if len(result.scratch_found) <= 8
               else f" … (+{len(result.scratch_found) - 8} more)"))

    if result.unreadable and not spec.skip_unreadable:
        raise UnreadableArtifactError(
            f"{len(result.unreadable)} covered file(s) could not be read: "
            + "; ".join(result.unreadable[:8])
            + ". A manifest that silently omits them is the M38 failure mode "
              "wearing a different hat; pass skip_unreadable=True to proceed "
              "with the omission REPORTED in the result")

    if result.broken_symlinks:
        message = (f"{len(result.broken_symlinks)} broken symlink(s) under "
                   f"{tree} — omitted, exactly as `find -L -type f` omits them, "
                   f"and reported because a silently narrowed manifest is the "
                   f"rake this module exists for: "
                   + "; ".join(result.broken_symlinks[:8]))
        if spec.strict_broken_symlinks:
            raise ManifestGeneratorError(message)
        result.findings.append(message)

    #  Rake M42(b) at generation grain: an artifact rewritten while we hashed it
    #  carries a digest of a state that no longer exists, and the manifest reads
    #  as tampered forever after.
    for row in candidates:
        try:
            stat = Path(row.walked_path).stat()
        except OSError as exc:
            result.changed_during_walk.append(f"{row.name} (vanished: {exc})")
            continue
        if (stat.st_size, stat.st_mtime_ns) != (row.size, row.mtime_ns):
            result.changed_during_walk.append(
                f"{row.name} (size/mtime moved under the walk)")
    if result.changed_during_walk and not spec.allow_concurrent_writes:
        raise ManifestGeneratorError(
            f"{len(result.changed_during_walk)} covered artifact(s) changed "
            f"DURING the walk — their digests are of states already gone, and a "
            f"manifest carrying them makes an honest artifact read as tampered "
            f"(rake M42(b)). Re-run when the producers are quiet: "
            + "; ".join(result.changed_during_walk[:8]))

    candidates.sort(key=lambda row: _sort_key(spec.sort)(row.name))
    result.rows = candidates

    #  ---- the M46 zero-assert, BEFORE any write ---------------------------
    patterns = spec.sanitization_patterns
    hits = scan_paths([row.name for row in result.rows], patterns)
    if spec.scan_bytes:
        byte_hits, not_scanned = scan_bytes(result.rows, patterns,
                                            spec.bytes_scan_max)
        hits.extend(byte_hits)
        result.bytes_not_scanned = not_scanned
        if not_scanned:
            result.findings.append(
                f"{len(not_scanned)} covered file(s) were NOT byte-scanned "
                f"(size cap or unreadable) — reported, never counted as clean: "
                + "; ".join(not_scanned[:6]))
    result.sanitization_hits = hits
    if hits:
        if spec.sanitize_mode == "refuse":
            assert_no_sanitization_hits(hits)
        hit_names = {hit.row_name for hit in hits}
        full_rows = list(result.rows)
        result.rows = [row for row in result.rows if row.name not in hit_names]
        for name in sorted(hit_names):
            result.excluded[name] = "sanitization hit (rake M46, excluded)"
        result.findings.append(
            f"{len(hit_names)} row(s) EXCLUDED by the sanitization sweep (rake "
            f"M46); the complete manifest is written to "
            f"{spec.unsanitized_copy} and stays desk-side per M46(c)")
    else:
        full_rows = list(result.rows)

    if spec.expect_rows is not None and len(result.rows) != spec.expect_rows:
        raise RowCountMismatchError(
            f"row count {len(result.rows)} != expected {spec.expect_rows} — rake "
            f"M43(b): the row-count-versus-derivation check is mandatory before "
            f"any regenerated manifest is promoted to head, and it is exactly "
            f"what caught the scratch row (2919 != 2918). Nothing was written")

    if result.empty_file_rows:
        result.findings.append(
            f"{len(result.empty_file_rows)} row(s) carry the sha256 of the EMPTY "
            f"string ({EMPTY_FILE_SHA256[:12]}…) — rake M43(c), worth a scan on "
            f"sight: " + "; ".join(result.empty_file_rows[:8]))

    if spec.compare_previous is not None and spec.compare_previous.is_file():
        diff = compare_to_previous(spec.compare_previous, result.rows)
        result.diff = diff
        if diff.shrank:
            result.findings.append(
                f"ROW COUNT SHRANK: {diff.n_previous} → {diff.n_current} "
                f"({diff.delta}). Rake M38(b): after any /models relocation a "
                f"shorter manifest is THIS RAKE — a walk that did not follow the "
                f"symlink — before it is a deletion. {len(diff.removed)} row(s) "
                f"absent: " + "; ".join(diff.removed[:6]))

    text = build_manifest_text([(row.digest, row.name) for row in result.rows],
                               headers=spec.headers,
                               escape_odd_names=spec.escape_odd_names)
    if spec.dry_run:
        result.findings.append("DRY RUN: nothing written")
        return result

    if spec.sanitize_mode == "exclude" and spec.unsanitized_copy is not None:
        full_text = build_manifest_text(
            [(row.digest, row.name) for row in full_rows],
            headers=list(spec.headers) + [
                "UNSANITIZED FULL COPY — desk-side only, never published "
                "(rake M46(c))"],
            escape_odd_names=spec.escape_odd_names)
        _atomic_write(full_text, spec.unsanitized_copy, tree)
        result.unsanitized_copy_written = str(spec.unsanitized_copy)

    result.n_bytes_written = _atomic_write(text, output, tree)
    result.written = True

    #  ---- M42, on the artifact rather than on the intent ------------------
    written = parse_manifest(output)
    if written.lists_itself:                 # pragma: no cover — unreachable
        raise ManifestGeneratorError(
            f"{output} lists itself after generation ({written.self_rows}) — "
            f"rake M42. The identity-based exclusion failed; this manifest can "
            f"never verify clean and must not be promoted")
    if len(written.rows) != len(result.rows):    # pragma: no cover — torn write
        raise ManifestGeneratorError(
            f"{output}: re-parsed {len(written.rows)} rows, generated "
            f"{len(result.rows)} — the artifact does not match the plan")
    return result


# ---------------------------------------------------------------- verify mode
def verify_manifest(manifest: Path, anchor: Optional[Path] = None,
                    quiet: bool = False) -> VerifyResult:
    """`sha256sum --check` semantics in-process, plus the M42 diagnosis it lacks.

    The exit-code split is the point (rake M42(c)): a digest mismatch is a DATA
    question and a self-row is a GENERATOR defect, so `data_clean` and
    `manifest_clean` are separate verdicts and a self-listing manifest over
    intact artifacts exits 3, not 1. A checker that conflates them blames
    innocent artifacts, which is what cost the ρ-of-record run its detour.
    """
    parsed = parse_manifest(manifest)
    base = anchor if anchor is not None else Path.cwd()
    result = VerifyResult(manifest_path=str(manifest), anchor=str(base),
                          self_rows=list(parsed.self_rows),
                          unparsed_lines=list(parsed.unparsed_lines))
    if parsed.self_rows:
        result.findings.append(
            f"SELF-LISTING (rake M42): {len(parsed.self_rows)} row(s) name the "
            f"manifest itself — {parsed.self_rows}. A file cannot contain its "
            f"own hash, so this manifest can never verify clean. The row is "
            f"EXCLUDED from the data verdict: the generator is at fault and the "
            f"artifacts are innocent")
        for raw in parsed.self_rows:
            result.rows.append(VerifyRow(
                name=raw, status="SELF-ROW", expected="",
                detail="generator defect, excluded from the data verdict"))
    if parsed.unparsed_lines:
        result.findings.append(
            f"UNPARSED LINE(S): {parsed.unparsed_lines[:6]} — reported, never "
            f"silently dropped: a line a parser cannot read may be the row a "
            f"checker needed")
    for row in parsed.rows:
        if row.is_self_row:
            continue
        artifact = base / row.name
        try:
            if not artifact.exists():
                result.rows.append(VerifyRow(name=row.name, status="MISSING",
                                             expected=row.digest,
                                             detail="no such file"))
                continue
            computed = sha256_of_file(artifact)
        except OSError as exc:
            result.rows.append(VerifyRow(name=row.name, status="MISSING",
                                         expected=row.digest,
                                         detail=f"unreadable: {exc}"))
            continue
        status: Literal["OK", "FAILED"] = (
            "OK" if computed == row.digest else "FAILED")
        result.rows.append(VerifyRow(name=row.name, status=status,
                                     expected=row.digest, computed=computed))
        if not quiet or status != "OK":
            print(f"{row.name}: {status}")
    #  Rake M15: a mass non-OK result means CHECK THE DIRECTORY FIRST. A verify
    #  where almost nothing resolves is an anchor question, not a data loss.
    covered = [r for r in result.rows if r.status != "SELF-ROW"]
    if covered and len(result.missing) > 0.5 * len(covered):
        result.findings.append(
            f"ANCHOR SUSPECT (rake M15): {len(result.missing)}/{len(covered)} "
            f"rows do not resolve under {base}. `sha256sum --check` must run "
            f"from the manifest's own anchor and that is NOT always its parent — "
            f"`manifests/outputs.sha256` lists `outputs/…` relative to the repo "
            f"root. Check the directory before investigating corruption")
    return result


# ------------------------------------------------------------------ selftest
def _raises(fn: Callable[[], Any], exc: type[BaseException]) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:                        # noqa: BLE001 — wrong class
        return False
    return False


def _ok(fn: Callable[[], Any]) -> bool:
    try:
        fn()
        return True
    except Exception:                        # noqa: BLE001
        return False


def _write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _spec(tree: Path, output: Path, **kwargs: Any) -> ManifestSpec:
    kwargs.setdefault("anchor", tree.parent)
    return ManifestSpec(tree=tree, output=output, **kwargs)


def selftest() -> int:                       # noqa: C901 — a checklist
    """CPU-only, data-independent verification of every rule this module encodes.

    Rake M44: every fixture is built inside a `TemporaryDirectory`, so nothing
    here resolves a banked artifact relative to cwd and the {data tree, no data
    tree} axis is VACUOUS — asserted below rather than assumed. The two live axes
    ({`sha256sum` present, absent} and {`read_composed_predictions` importable,
    not}) each degrade to a named skip counted in the tail.
    """
    import tempfile as _tempfile

    checks: list[tuple[str, bool, str]] = []
    skips: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)

    def skip(name: str, why: str) -> None:
        """A NAMED skip (rake M44): a block that cannot run in THIS
        configuration. Recorded as run-and-absent, counted in the tail, and
        NEVER a non-zero exit — nothing was asserted and found wanting."""
        skips.append(name)
        checks.append((f"SKIPPED: {name}", True, why))
        logger.info("SKIP %s — %s", name, why)

    sha256sum_bin = shutil.which("sha256sum")

    # ---- 1. byte compatibility with GNU sha256sum ---------------------------
    print("== selftest 1: the row format is byte-identical to `sha256sum` ==")
    with _tempfile.TemporaryDirectory(prefix="mgen_bytes_") as td:
        base = Path(td)
        tree = base / "store"
        for name, body in (("a.txt", "alpha\n"), ("sub/b.bin", "beta\n"),
                           ("sub/deep/c.json", '{"k": 1}\n')):
            _write(tree / name, body)
        out = base / "MANIFEST-store.sha256"
        built = generate_tree_manifest(_spec(tree, out))
        text = out.read_text()
        check("rows are `<64 hex><two spaces><path>`",
              all(re.match(r"^[0-9a-f]{64} {2}\S", line)
                  for line in text.splitlines()),
              f"{built.n_rows} rows")
        check("row names are relative to the ANCHOR, not to the tree",
              [r.name for r in built.rows] == ["store/a.txt", "store/sub/b.bin",
                                               "store/sub/deep/c.json"],
              str([r.name for r in built.rows]))
        check("the manifest ends with a newline, like `sha256sum`'s output",
              text.endswith("\n"))
        if sha256sum_bin is None:
            skip("sha256sum-byte-compat",
                 "no `sha256sum` on PATH: the byte-identity and `--check` "
                 "cross-checks compare against the real GNU binary and cannot "
                 "run in this configuration")
        else:
            proc = subprocess.run(
                [sha256sum_bin, *[r.name for r in built.rows]],
                cwd=base, capture_output=True, text=True, check=True)
            check("output is BYTE-IDENTICAL to what GNU `sha256sum` writes",
                  proc.stdout == text,
                  f"{len(text)} B ours vs {len(proc.stdout)} B GNU's")
            checked = subprocess.run([sha256sum_bin, "--check", "--quiet",
                                      str(out)], cwd=base, capture_output=True,
                                     text=True)
            check("`sha256sum --check --quiet` verifies it clean, rc=0",
                  checked.returncode == 0,
                  f"rc={checked.returncode} {checked.stderr.strip()}")
            #  `#` headers: `manifests/collection.sha256` carries three, so the
            #  claim that they are check-safe is measured, never assumed.
            out_h = base / "MANIFEST-headers.sha256"
            generate_tree_manifest(_spec(tree, out_h, headers=[
                "a header line", "generated by the selftest"]))
            checked_h = subprocess.run([sha256sum_bin, "--check", "--quiet",
                                        str(out_h)], cwd=base,
                                       capture_output=True, text=True)
            check("`#` header lines are ignored by `sha256sum --check` (rc=0)",
                  checked_h.returncode == 0
                  and out_h.read_text().startswith("# a header line\n"),
                  f"rc={checked_h.returncode}")
        check("a digest that is not 64 lowercase hex is refused (rake M40)",
              _raises(lambda: format_manifest_row("abc", "x"),
                      ManifestFormatError))
        check("a control character in a path is refused unconditionally",
              _raises(lambda: format_manifest_row("0" * 64, "a\tb"),
                      ManifestFormatError))
        check("a backslash in a path is refused by DEFAULT",
              _raises(lambda: format_manifest_row("0" * 64, "back\\slash"),
                      ManifestFormatError))
        check("…and escapes to GNU's form when explicitly allowed",
              format_manifest_row("0" * 64, "back\\slash", escape_odd_names=True)
              == "\\" + "0" * 64 + "  back\\\\slash",
              "leading `\\`, backslash doubled — coreutils 9.5's exact shape")
        check("the escape round-trips through the parser",
              _unescape_name("back\\\\slash") == "back\\slash"
              and _unescape_name("two\\nlines") == "two\nlines")

    # ---- 2. RAKE M38: symlinks, symlinked dirs, and cycles ------------------
    print("== selftest 2: RAKE M38 — `find -L`, symlinked trees, link cycles ==")
    with _tempfile.TemporaryDirectory(prefix="mgen_m38_") as td:
        base = Path(td)
        tree = base / "store"
        _write(tree / "local.txt", "local\n")
        #  The /models placement pattern: the real bytes live elsewhere and the
        #  tree reaches them through a symlink.
        relocated = base / "models" / "relocated"
        _write(relocated / "banked.npz", "banked\n")
        _write(relocated / "nested" / "deep.json", "{}\n")
        (tree / "relocated").symlink_to(relocated, target_is_directory=True)
        loose = base / "models" / "loose.bin"
        _write(loose, "loose\n")
        (tree / "loose.bin").symlink_to(loose)

        naive = sorted(p.name for p in tree.rglob("*") if p.is_file())
        out = base / "MANIFEST-m38.sha256"
        built = generate_tree_manifest(_spec(tree, out))
        names = [r.name for r in built.rows]
        check("the walk finds the symlinked FILE and the symlinked DIRECTORY's "
              "contents — the rows `find -type f` drops",
              names == ["store/local.txt", "store/loose.bin",
                        "store/relocated/banked.npz",
                        "store/relocated/nested/deep.json"], str(names))
        check("a naive `rglob` walk finds FEWER: this is the rake, measured",
              len(naive) < len(names), f"rglob {len(naive)} vs find -L {len(names)}")
        check("symlink-reached rows are LABELLED, so a relocation is visible",
              sorted(built.symlinked_rows) == ["store/loose.bin",
                                               "store/relocated/banked.npz",
                                               "store/relocated/nested/deep.json"],
              str(sorted(built.symlinked_rows)))
        #  M38(a): a cycle must terminate.
        (tree / "relocated" / "loop").symlink_to(tree, target_is_directory=True)
        looped = generate_tree_manifest(_spec(tree, base / "MANIFEST-loop.sha256"))
        check("a symlink CYCLE back to the root terminates and does not "
              "double-count (the (st_dev, st_ino) visit guard)",
              [r.name for r in looped.rows] == names, str(len(looped.rows)))
        (tree / "relocated" / "loop").unlink()
        #  Broken symlinks: `find -L -type f` omits them; silence is the rake.
        (tree / "dangling.bin").symlink_to(base / "nowhere" / "gone.bin")
        with_broken = generate_tree_manifest(
            _spec(tree, base / "MANIFEST-broken.sha256"))
        check("a BROKEN symlink is omitted (as `find -L -type f` omits it) and "
              "REPORTED, never silently narrowing the manifest",
              with_broken.broken_symlinks == ["store/dangling.bin"]
              and any("broken symlink" in f for f in with_broken.findings),
              str(with_broken.broken_symlinks))
        check("…and strict mode makes it a HALT",
              _raises(lambda: generate_tree_manifest(
                  _spec(tree, base / "MANIFEST-strict.sha256",
                        strict_broken_symlinks=True)), ManifestGeneratorError))
        (tree / "dangling.bin").unlink()
        #  M38(b): a shrunk manifest after a relocation is the rake, not deletion.
        previous = base / "PREV.sha256"
        previous.write_text(out.read_text() + f"{'0' * 64}  store/vanished.bin\n")
        shrunk = generate_tree_manifest(
            _spec(tree, base / "MANIFEST-shrunk.sha256",
                  compare_previous=previous))
        check("a SHRUNK row count is reported as rake M38(b) first, deletion "
              "second",
              shrunk.diff is not None and shrunk.diff.shrank
              and any("M38(b)" in f for f in shrunk.findings),
              f"{shrunk.diff.n_previous} → {shrunk.diff.n_current}"
              if shrunk.diff else "")
        check("the diff names added/removed/changed rows, not just a count",
              shrunk.diff is not None
              and shrunk.diff.removed == ["store/vanished.bin"]
              and not shrunk.diff.added)

    # ---- 3. RAKE M42: a manifest can never list itself ----------------------
    print("== selftest 3: RAKE M42 — self-listing, by identity not by glob ==")
    with _tempfile.TemporaryDirectory(prefix="mgen_m42_") as td:
        base = Path(td)
        tree = base / "store"
        _write(tree / "a.txt", "a\n")
        _write(tree / "b.txt", "b\n")
        #  The output INSIDE the tree it hashes — the per-staging MANIFEST case.
        #  Generated TWICE on purpose: the second run is the one where the
        #  manifest already exists on disk and the walk must refuse to hash it.
        inside = tree / "MANIFEST-store.sha256"
        generate_tree_manifest(_spec(tree, inside))
        built = generate_tree_manifest(_spec(tree, inside))
        check("a manifest written INSIDE its own tree does not list itself, "
              "even on the regeneration where it is already on disk",
              [r.name for r in built.rows] == ["store/a.txt", "store/b.txt"]
              and "store/MANIFEST-store.sha256" in built.excluded,
              built.excluded.get("store/MANIFEST-store.sha256", ""))
        before = inside.read_bytes()
        generate_tree_manifest(_spec(tree, inside))
        check("a regeneration is IDEMPOTENT: byte-identical head, run over run",
              inside.read_bytes() == before, f"{len(before)} B")
        verified = verify_manifest(inside, anchor=base, quiet=True)
        check("…and it verifies clean, rc=0, with no self-row to adjudicate",
              verified.data_clean and verified.manifest_clean
              and verified.exit_code() == 0, f"{verified.n_ok}/2 OK")
        #  BY IDENTITY: an output whose name matches no exclusion glob at all.
        #  A glob-only exclusion (`-not -name "MANIFEST-*"`, the convention M42
        #  names) would list THIS one, which is the whole reason the check here
        #  is on resolved paths.
        odd = tree / "coverage-index.txt"
        generate_tree_manifest(_spec(tree, odd))
        built_odd = generate_tree_manifest(_spec(tree, odd))
        check("the exclusion is by IDENTITY, so an output named nothing like "
              "`MANIFEST-*` still cannot list itself",
              "store/coverage-index.txt" in built_odd.excluded
              and "rake M42" in built_odd.excluded["store/coverage-index.txt"]
              and not any(fnmatch.fnmatch("coverage-index.txt", g)
                          for g in DEFAULT_EXCLUDE_GLOBS),
              built_odd.excluded.get("store/coverage-index.txt", ""))
        check("…and that output verifies clean too (nothing to adjudicate)",
              verify_manifest(odd, anchor=base, quiet=True).exit_code() == 0)
        odd.unlink()
        #  The checker half: a hand-built self-listing manifest is DIAGNOSED.
        selfy = base / "MANIFEST-selfy.sha256"
        rows = "".join(f"{sha256_of_file(tree / n)}  store/{n}\n"
                       for n in ("a.txt", "b.txt"))
        selfy.write_text(rows + f"{'0' * 64}  MANIFEST-selfy.sha256\n")
        parsed = parse_manifest(selfy)
        check("a self-row is DETECTED and excluded from `by_name`",
              parsed.lists_itself and len(parsed.by_name) == 2,
              str(parsed.self_rows))
        result = verify_manifest(selfy, anchor=base, quiet=True)
        check("verify calls it a GENERATOR defect with the DATA still clean — "
              "rake M42(c), the artifacts are innocent",
              result.data_clean and not result.manifest_clean
              and result.exit_code() == 3
              and any("SELF-LISTING (rake M42)" in f for f in result.findings),
              f"exit code {result.exit_code()} (3 = manifest defect, data clean)")
        #  M42(d): both banked dialects, and a two-word comment is not a row.
        dotted = base / "MANIFEST-dotted.sha256"
        dotted.write_text(
            "# a node-side manifest\n# phantom\n"
            + rows.replace("  store/", "  ./store/"))
        parsed_dot = parse_manifest(dotted)
        check("BOTH dialects parse to the same keys, and a two-word `#` comment "
              "does not become a phantom row (rake M42(d))",
              parsed_dot.dialect == "dot-prefixed"
              and parsed_dot.by_name == parsed.by_name
              and len(parsed_dot.header_lines) == 2,
              f"{parsed_dot.dialect}, {len(parsed_dot.header_lines)} headers")
        check("a duplicate row name is a HALT, never a silent collapse (M18)",
              _raises(lambda: parse_manifest(_write(
                  base / "dupe.sha256", rows + rows.splitlines()[0] + "\n")),
                  ManifestFormatError))

    # ---- 4. RAKE M43: scratch never inside the hashed tree ------------------
    print("== selftest 4: RAKE M43 — scratch outside the tree, row counts ==")
    with _tempfile.TemporaryDirectory(prefix="mgen_m43_") as td:
        base = Path(td)
        tree = base / "store"
        _write(tree / "a.txt", "a\n")
        _write(tree / "b.txt", "b\n")
        out = base / "MANIFEST-m43.sha256"
        scratch = _write(tree / "MANIFEST-store.sha256.tmp", "")
        check("a tree containing scratch is REFUSED, not silently swept",
              _raises(lambda: generate_tree_manifest(_spec(tree, out)),
                      ScratchInTreeError))
        check("the refusal names the offending file and the glob it matched",
              "MANIFEST-store.sha256.tmp" in str(
                  _raises_message(lambda: generate_tree_manifest(_spec(tree, out)))))
        check("the DEFAULT `MANIFEST-*` exclusion does NOT silently swallow a "
              "`MANIFEST-….sha256.tmp`: scratch is checked FIRST, or a default "
              "reports a clean run over a tree that is mid-write",
              fnmatch.fnmatch("MANIFEST-store.sha256.tmp",
                              DEFAULT_EXCLUDE_GLOBS[0])
              and _raises(lambda: generate_tree_manifest(_spec(tree, out)),
                          ScratchInTreeError),
              "the name matches the default exclusion AND still refuses — this "
              "ordering bug was live until this selftest caught it")
        allowed = generate_tree_manifest(_spec(
            tree, out, exclude_globs=[*DEFAULT_EXCLUDE_GLOBS, "*.sha256.tmp"]))
        check("…and proceeds once the scratch is EXPLICITLY excluded by a glob "
              "the CALLER supplied",
              [r.name for r in allowed.rows] == ["store/a.txt", "store/b.txt"]
              and "EXPLICITLY excluded" in allowed.excluded[
                  "store/MANIFEST-store.sha256.tmp"],
              allowed.excluded["store/MANIFEST-store.sha256.tmp"])
        scratch.unlink()
        #  The structural defence: no file at all appears inside the tree while
        #  the manifest is being produced. Proven by watching the tree from a
        #  hash callback rather than by reading the code.
        seen_during: list[list[str]] = []
        real_hash = sha256_of_file

        def _watching(path: Path) -> str:
            seen_during.append(sorted(p.name for p in tree.rglob("*")))
            return real_hash(path)

        globals()["sha256_of_file"] = _watching
        try:
            generate_tree_manifest(_spec(tree, tree / "MANIFEST-inside.sha256"))
        finally:
            globals()["sha256_of_file"] = real_hash
        check("NOTHING is written inside the tree while it is being hashed — "
              "the M43 self-inclusion cannot occur even if staging were wrong",
              all("MANIFEST-inside.sha256" not in snapshot
                  for snapshot in seen_during),
              f"{len(seen_during)} snapshot(s) taken mid-walk")
        staging = scratch_staging_dir(tree, tree / "MANIFEST-inside.sha256")
        check("the atomic-rename staging dir is PROVEN outside the hashed tree",
              not _is_within(staging, tree), str(staging))
        check("…even when the output itself lives inside the tree",
              not _is_within(scratch_staging_dir(tree, tree / "x.sha256"), tree))
        #  M43(b): the row-count-versus-derivation check that caught the original.
        check("a row count that disagrees with the caller's derivation HALTs "
              "before anything is promoted (rake M43(b))",
              _raises(lambda: generate_tree_manifest(
                  _spec(tree, out, expect_rows=99)), RowCountMismatchError))
        check("…and the matching count proceeds",
              generate_tree_manifest(_spec(tree, out, expect_rows=2)).written)
        #  M43(c): the empty-file sha is a red flag worth a scan on sight.
        _write(tree / "empty.bin", "")
        with_empty = generate_tree_manifest(_spec(tree, out))
        check("a row carrying the sha256 of the EMPTY string is flagged (M43(c))",
              with_empty.empty_file_rows == ["store/empty.bin"]
              and any(EMPTY_FILE_SHA256[:12] in f for f in with_empty.findings),
              EMPTY_FILE_SHA256[:16] + "…")
        check("the empty-string sha constant is the real one",
              hashlib.sha256(b"").hexdigest() == EMPTY_FILE_SHA256)

    # ---- 5. RAKE M46: sanitization, filenames included ----------------------
    print("== selftest 5: RAKE M46 — the blocking zero-assert over PATHS ==")
    with _tempfile.TemporaryDirectory(prefix="mgen_m46_") as td:
        base = Path(td)
        tree = base / "store"
        _write(tree / "clean.json", '{"ok": 1}\n')
        _write(tree / "provenance-secretalias-01.json", '{"host": "x"}\n')
        out = base / "MANIFEST-m46.sha256"
        patterns = [SanitizationPattern(name="alias", pattern="secretalias")]
        check("a hit in a FILENAME blocks the write — rake M46(b), filenames "
              "are content",
              _raises(lambda: generate_tree_manifest(
                  _spec(tree, out, sanitization_patterns=patterns)),
                  SanitizationLeakError) and not out.exists(),
              "and the manifest does NOT exist afterwards")
        hits = scan_paths(["a/secretalias/b.json", "clean.json"], patterns)
        check("the sweep is a ZERO-ASSERT: any hit is the failure, and the "
              "count is never the verdict (`grep -c` exits 0 when it matches)",
              len(hits) == 1 and _raises(
                  lambda: assert_no_sanitization_hits(hits),
                  SanitizationLeakError)
              and _ok(lambda: assert_no_sanitization_hits([])))
        check("a hit is reported REDACTED by default, so the refusal message is "
              "not itself a place the string propagates to",
              "⟨REDACTED⟩" in hits[0].redacted
              and "secretalias" not in hits[0].redacted
              and hits[0].matched is None, hits[0].redacted)
        check("…and the raw match is available only when explicitly asked for",
              scan_paths(["a/secretalias/b"], patterns,
                         show_matches=True)[0].matched == "secretalias")
        #  M46(c): the exclude lane REQUIRES a desk-side full copy.
        check("sanitize_mode='exclude' without an unsanitized copy is refused "
              "at construction (rake M46(c))",
              _raises(lambda: ManifestSpec(tree=tree, output=out,
                                           sanitize_mode="exclude"), ValueError))
        full = base / "full-unsanitized.sha256"
        excluded = generate_tree_manifest(_spec(
            tree, out, sanitization_patterns=patterns,
            sanitize_mode="exclude", unsanitized_copy=full))
        published = [r.name for r in excluded.rows]
        check("the exclude lane drops the matching row from the PUBLISHED "
              "manifest",
              published == ["store/clean.json"], str(published))
        check("…and writes the COMPLETE manifest desk-side, so coverage stays "
              "honest",
              full.is_file() and len(parse_manifest(full).rows) == 2
              and excluded.unsanitized_copy_written == str(full),
              f"{len(parse_manifest(full).rows)} rows in the full copy")
        check("the published manifest carries no trace of the excluded name",
              "secretalias" not in out.read_text())
        #  bytes half
        _write(tree / "notes.txt", "harmless\nan embedded secretalias here\n")
        byte_patterns = [SanitizationPattern(name="alias", pattern="secretalias",
                                             paths=True, contents=True)]
        check("bytes are scanned too when asked, and block just as loudly",
              _raises(lambda: generate_tree_manifest(
                  _spec(tree, base / "M2.sha256",
                        sanitization_patterns=byte_patterns, scan_bytes=True)),
                  SanitizationLeakError))
        big = _write(tree / "big.bin", "x" * 4096)
        _, not_scanned = scan_bytes(
            [HashedFile(name="store/big.bin", digest="0" * 64, size=4096,
                        mtime_ns=0, walked_path=str(big))],
            byte_patterns, max_size=100)
        check("a file above the byte-scan cap is REPORTED as not scanned, never "
              "counted as clean",
              len(not_scanned) == 1 and "cap" in not_scanned[0], not_scanned[0])
        #  The generic set must not cry wolf: both of these over-matched until
        #  the CLI smoke test found them, and a sweep that fires on honest
        #  artifacts is a sweep somebody turns off.
        generic = generic_patterns()
        check("the generic set fires on a REAL absolute home path in a row name",
              len(scan_paths(["/home/someone/notes.txt"], generic)) == 1)
        check("…and NOT on a relative row that merely has a directory called "
              "`home` in it",
              scan_paths(["outputs/home/notes.txt",
                          "store/home/copied-notes.txt"], generic) == [],
              "the /home must start the string or follow a non-filename char")
        check("it fires on a real e-mail address",
              len(scan_paths(["contact-someone@example.com.json"],
                             generic)) == 1)
        check("…and NOT on a versioned artifact name like `bank@v1.2.json`",
              scan_paths(["outputs/banks/bank@v1.2.json",
                          "outputs/x/states@v2.1.npz"], generic) == [])
        #  The guard against this file ever carrying a campaign alias is a
        #  FROZEN ROSTER of pattern classes, not a regex that would have to
        #  spell the aliases out to look for them. Adding one here fails here.
        check("no campaign-specific pattern can be compiled into this module: "
              "the shipped roster is frozen and generic by construction",
              tuple(GENERIC_PATH_PATTERNS) == GENERIC_PATTERN_NAMES
              and {p.name for p in generic_patterns()}
              == set(GENERIC_PATTERN_NAMES)
              and "patterns-file" in (__doc__ or ""),
              f"{len(GENERIC_PATH_PATTERNS)} generic classes shipped; campaign "
              f"patterns load from --patterns-file")
        check("an absent pattern file is a HALT, because an empty sweep is the "
              "most convincing false negative available",
              _raises(lambda: load_patterns_file(base / "nope.txt"),
                      ManifestGeneratorError))
        pattern_file = _write(base / "patterns.txt",
                              "# desk-side\nalias = secretalias\n"
                              "contents: creds = password\\s*=\\s*\\S+\n"
                              "bare-regex-[0-9]+\n")
        loaded = load_patterns_file(pattern_file)
        check("the pattern file parses named, scoped and bare forms",
              [p.name for p in loaded] == ["alias", "creds", "pattern-4"]
              and loaded[1].contents and not loaded[1].paths
              and loaded[2].paths, str([p.name for p in loaded]))
        check("an invalid regex in the file is a HALT with the line number",
              _raises(lambda: load_patterns_file(
                  _write(base / "bad.txt", "broken = [unclosed\n")),
                  ManifestGeneratorError))

    # ---- 6. determinism, ordering, and the concurrency guard ---------------
    print("== selftest 6: deterministic ordering, idempotence, live writes ==")
    with _tempfile.TemporaryDirectory(prefix="mgen_order_") as td:
        base = Path(td)
        tree = base / "store"
        for name in ("z.txt", "a.txt", "m/b.txt", "M/c.txt", "a_b.txt", "a-b.txt"):
            _write(tree / name, name)
        out = base / "MANIFEST-order.sha256"
        first = generate_tree_manifest(_spec(tree, out))
        second = generate_tree_manifest(_spec(tree, out))
        check("two runs over an unchanged tree produce IDENTICAL bytes",
              [r.name for r in first.rows] == [r.name for r in second.rows]
              and first.manifest_text() == second.manifest_text(),
              f"{first.n_rows} rows")
        names = [r.name for r in first.rows]
        check("the default order is BYTE-wise (`LC_ALL=C sort`), the only "
              "ordering that reproduces on another machine",
              names == sorted(names, key=lambda s: s.encode()), str(names[:3]))
        if sha256sum_bin is None:
            skip("gnu-sort-byte-order-agreement",
                 "no `sha256sum` on PATH; the LC_ALL=C `find | sort | xargs` "
                 "pipeline cross-check needs the real binaries")
        else:
            pipeline = subprocess.run(
                ["bash", "-c",
                 "find -L store -type f -print0 | LC_ALL=C sort -z | "
                 "xargs -0 sha256sum"],
                cwd=base, capture_output=True, text=True, check=True)
            check("byte order reproduces the banked `find -L | LC_ALL=C sort | "
                  "xargs sha256sum` pipeline EXACTLY",
                  pipeline.stdout == out.read_text(),
                  f"{len(pipeline.stdout.splitlines())} pipeline rows")
        check("locale ordering is offered and labelled non-portable",
              generate_tree_manifest(
                  _spec(tree, base / "MANIFEST-loc.sha256", sort="locale")
              ).sort == "locale"
              and "nobody's default" in (__doc__ or ""))
        #  The concurrency guard: an artifact rewritten under the walk.
        real_hash = sha256_of_file

        def _rewriting(path: Path) -> str:
            digest = real_hash(path)
            if path.name == "z.txt":
                path.write_text("rewritten while the walk was reading it\n")
                os.utime(path, (0, 0))
            return digest

        globals()["sha256_of_file"] = _rewriting
        try:
            check("an artifact rewritten DURING the walk is a HALT: its digest "
                  "is of a state already gone (rake M42(b) at generation grain)",
                  _raises(lambda: generate_tree_manifest(
                      _spec(tree, base / "MANIFEST-race.sha256")),
                      ManifestGeneratorError))
        finally:
            globals()["sha256_of_file"] = real_hash
        #  Error handling: an anchor that is not an ancestor, an absent tree.
        check("an anchor that is not an ancestor of the tree is refused (no row "
              "may carry a `../` escape)",
              _raises(lambda: generate_tree_manifest(ManifestSpec(
                  tree=tree, output=out, anchor=tree / "m")),
                  ManifestGeneratorError))
        check("an absent tree is refused, never written as an empty manifest",
              _raises(lambda: generate_tree_manifest(
                  _spec(base / "gone", base / "x.sha256")),
                  ManifestGeneratorError))
        check("a dry run reports the full plan and writes NOTHING",
              (lambda r: r.n_rows == first.n_rows and not r.written
               and not (base / "dry.sha256").exists())(
                  generate_tree_manifest(_spec(tree, base / "dry.sha256",
                                               dry_run=True))))

    # ---- 7. verify mode: sha256sum --check semantics ------------------------
    print("== selftest 7: verify mode — `sha256sum --check`, in process ==")
    with _tempfile.TemporaryDirectory(prefix="mgen_verify_") as td:
        base = Path(td)
        tree = base / "store"
        _write(tree / "keep.txt", "keep\n")
        _write(tree / "tamper.txt", "before\n")
        _write(tree / "vanish.txt", "here\n")
        out = base / "MANIFEST-verify.sha256"
        generate_tree_manifest(_spec(tree, out))
        clean = verify_manifest(out, anchor=base, quiet=True)
        check("a fresh manifest verifies OK on every row, exit code 0",
              clean.data_clean and clean.n_ok == 3 and clean.exit_code() == 0,
              f"{clean.n_ok}/3 OK")
        (tree / "tamper.txt").write_text("after\n")
        (tree / "vanish.txt").unlink()
        dirty = verify_manifest(out, anchor=base, quiet=True)
        check("a changed artifact reads FAILED and a deleted one MISSING — "
              "distinct statuses, because they are distinct findings",
              [r.name for r in dirty.failed] == ["store/tamper.txt"]
              and [r.name for r in dirty.missing] == ["store/vanish.txt"]
              and dirty.exit_code() == 1,
              f"{dirty.n_ok} OK, {len(dirty.failed)} FAILED, "
              f"{len(dirty.missing)} MISSING")
        if sha256sum_bin is None:
            skip("sha256sum-check-verdict-agreement",
                 "no `sha256sum` on PATH: the verdict-agreement cross-check "
                 "compares our statuses against the real GNU checker")
        else:
            gnu = subprocess.run([sha256sum_bin, "--check", str(out)], cwd=base,
                                 capture_output=True, text=True)
            gnu_status = {line.rsplit(": ", 1)[0]: line.rsplit(": ", 1)[1]
                          for line in gnu.stdout.splitlines() if ": " in line}
            ours = {r.name: r.status for r in dirty.rows}
            check("our per-row verdicts AGREE with GNU `sha256sum --check`",
                  gnu_status.get("store/keep.txt") == "OK"
                  and ours["store/keep.txt"] == "OK"
                  and gnu_status.get("store/tamper.txt") == "FAILED"
                  and ours["store/tamper.txt"] == "FAILED"
                  and gnu.returncode == 1 and dirty.exit_code() == 1,
                  f"GNU rc={gnu.returncode}, ours={dirty.exit_code()}")
        wrong = verify_manifest(out, anchor=tree, quiet=True)
        check("a verify from the WRONG anchor says ANCHOR SUSPECT (rake M15), "
              "not 'the data is gone'",
              any("ANCHOR SUSPECT" in f for f in wrong.findings),
              f"{len(wrong.missing)}/{len(wrong.rows)} unresolved")
        check("an absent manifest is a clear HALT, not an empty pass",
              _raises(lambda: verify_manifest(base / "nope.sha256"),
                      ManifestGeneratorError))
        weird = _write(base / "weird.sha256",
                       "not a manifest line at all\n"
                       + out.read_text().splitlines()[0] + "\n")
        parsed_weird = parse_manifest(weird)
        check("an unparsable line is reported, never silently dropped",
              parsed_weird.unparsed_lines and len(parsed_weird.rows) == 1,
              str(parsed_weird.unparsed_lines))

    # ---- 8. the M44 configuration statement, asserted -----------------------
    print("== selftest 8: rake M44 — the configuration this run measured ==")
    check("every fixture is tmpdir-based, so the {data tree, no data tree} axis "
          "is VACUOUS here (asserted, not assumed): no default resolves a "
          "banked path",
          ManifestSpec(tree=Path("."), output=Path("x.sha256")).anchor
          == Path.cwd()
          and "outputs" not in str(DEFAULT_EXCLUDE_GLOBS),
          "the only cwd-dependent default is the anchor, which every fixture "
          "passes explicitly")
    source = Path(__file__).read_text()
    check("this module imports nothing from the deep-learning stack (asserted "
          "against its own source), so the {torch, no torch} axis is vacuous "
          "too — it runs on the CPU spine of pyproject.toml",
          not re.search(r"^\s*(import|from)\s+(torch|transformers|accelerate)\b",
                        source, re.M),
          "stdlib + pydantic only — not even numpy")
    check("…and it imports no numpy/scipy either, so it deploys node-side with "
          "nothing but the interpreter and pydantic",
          not re.search(r"^\s*(import|from)\s+(numpy|scipy)\b", source, re.M))
    try:
        from metabasis.scripts import read_composed_predictions as _rcp
    except Exception as exc:                 # noqa: BLE001 — availability probe
        _rcp = None                          # type: ignore[assignment]
        skip("parser-agreement-with-read_composed_predictions",
             f"the banked manifest parser could not be imported in this "
             f"configuration ({type(exc).__name__}: {exc}); the cross-check that "
             f"the two parsers agree cannot run")
    if _rcp is not None:
        check("the MANIFEST-* exclusion convention has ONE value across the "
              "codebase (rake M26: a sweep can find every consumer)",
              _rcp.MANIFEST_NAME_PREFIX == MANIFEST_NAME_PREFIX,
              MANIFEST_NAME_PREFIX)
        with _tempfile.TemporaryDirectory(prefix="mgen_xcheck_") as td:
            base = Path(td)
            tree = base / "store"
            _write(tree / "a.txt", "a\n")
            _write(tree / "sub/b.txt", "b\n")
            out = base / "MANIFEST-x.sha256"
            generate_tree_manifest(_spec(tree, out, headers=["a header"]))
            theirs = _rcp.parse_sha256_manifest(out)
            ours = parse_manifest(out)
            check("the banked parser reads our output identically (same keys, "
                  "same digests, no phantom row from the `#` header)",
                  theirs.by_name == ours.by_name and theirs.n_header_lines == 1
                  and not theirs.unparsed_lines and not theirs.lists_itself,
                  f"{len(theirs.by_name)} rows agreed")
            audit = _rcp.audit_manifest(out, anchor=base)
            check("…and its M42 AUDIT finds a manifest generated here CLEAN",
                  audit.clean and not audit.findings,
                  str(audit.findings))
            check("our digests match the banked `sha256_of` helper byte for byte",
                  _rcp.sha256_of(tree / "a.txt")
                  == sha256_of_file(tree / "a.txt"))

    failures = [c for c in checks if not c[1]]
    print(f"\nselftest: {len(failures)} failure(s)")
    for name, _, detail in failures:
        print(f"  MISS {name} {detail}")
    # RAKE M44: coverage is part of the verdict, per configuration.
    print(f"selftest checks run: {len(checks)} ({len(skips)} named skip(s))")
    for name in skips:
        print(f"  SKIPPED {name}")
    print(f"configuration: python {sys.version.split()[0]} · "
          f"sha256sum {'present' if sha256sum_bin else 'ABSENT'} · "
          f"read_composed_predictions "
          f"{'importable' if _rcp is not None else 'NOT importable'} · "
          f"fixtures tmpdir-only (data tree irrelevant) · cwd {Path.cwd()}")
    return 1 if failures else 0


def _raises_message(fn: Callable[[], Any]) -> str:
    try:
        fn()
    except Exception as exc:                 # noqa: BLE001
        return str(exc)
    return ""


# ----------------------------------------------------------------------- CLI
def _print_build_report(result: ManifestBuildResult) -> None:
    written_note = "" if result.written else "  (NOT WRITTEN)"
    symlink_note = ""
    if result.symlinked_rows:
        symlink_note = (f" ({len(result.symlinked_rows)} reached through "
                        f"symlinks — rake M38)")
    print(f"tree           {result.tree}")
    print(f"anchor         {result.anchor}")
    print(f"manifest       {result.manifest_path}{written_note}")
    print(f"rows           {result.n_rows}{symlink_note}")
    print(f"order          {result.sort}-wise")
    if result.excluded:
        print(f"excluded       {len(result.excluded)}")
        for name, why in list(result.excluded.items())[:10]:
            print(f"  - {name}: {why}")
    if result.diff is not None:
        diff = result.diff
        print(f"vs previous    {diff.n_previous} → {diff.n_current} "
              f"({diff.delta:+d}); +{len(diff.added)} -{len(diff.removed)} "
              f"~{len(diff.changed)}")
    for finding in result.findings:
        print(f"FINDING        {finding}")
    if result.unsanitized_copy_written:
        print(f"full copy      {result.unsanitized_copy_written} (desk-side, "
              f"rake M46(c))")
    if result.written:
        print(f"bytes written  {result.n_bytes_written}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="The canonical sha256 manifest generator/verifier: one "
                    "shared lane with rakes M38/M42/M43/M46 enforced in code.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true",
                      help="CPU-only, data-independent verification")
    mode.add_argument("--tree", type=Path,
                      help="the tree to hash (rows are named relative to "
                           "--anchor)")
    mode.add_argument("--verify", type=Path, metavar="MANIFEST",
                      help="verify a manifest in process (`sha256sum --check` "
                           "semantics + the M42 diagnosis)")
    mode.add_argument("--scan-only", type=Path, metavar="MANIFEST",
                      help="run the M46 sanitization zero-assert over an "
                           "existing manifest's PATHS, as its own blocking "
                           "command")
    ap.add_argument("--output", type=Path, help="manifest path (with --tree)")
    ap.add_argument("--anchor", type=Path,
                    help="directory the rows are relative to (default: cwd)")
    ap.add_argument("--header", action="append", default=[], metavar="TEXT",
                    help="a `#` header line; repeatable")
    ap.add_argument("--exclude", action="append", default=None, metavar="GLOB",
                    help="basename glob to exclude; repeatable. Replaces the "
                         f"defaults {DEFAULT_EXCLUDE_GLOBS}")
    ap.add_argument("--exclude-path", action="append", default=[],
                    metavar="GLOB", help="row-name glob to exclude; repeatable")
    ap.add_argument("--patterns-file", type=Path,
                    help="desk-side sanitization patterns (never in this repo)")
    ap.add_argument("--generic-patterns", action="store_true",
                    help="also apply the generic-by-construction pattern set")
    ap.add_argument("--sanitize-mode", choices=("refuse", "exclude"),
                    default="refuse",
                    help="refuse (default, M46(a)) or exclude matching rows "
                         "(requires --unsanitized-copy, M46(c))")
    ap.add_argument("--unsanitized-copy", type=Path,
                    help="where the COMPLETE manifest goes in exclude mode")
    ap.add_argument("--scan-bytes", action="store_true",
                    help="also scan covered artifacts' bytes (capped)")
    ap.add_argument("--show-matches", action="store_true",
                    help="print matched text in findings (desk-side only)")
    ap.add_argument("--sort", choices=("byte", "locale"), default="byte",
                    help="byte (default, portable) or locale (reproduces the "
                         "historical `sort -z` order; NOT portable)")
    ap.add_argument("--expect-rows", type=int,
                    help="the caller's independent row-count derivation (M43(b))")
    ap.add_argument("--compare-previous", type=Path, metavar="MANIFEST",
                    help="diff against the previous manifest of record (M38(b))")
    ap.add_argument("--strict-broken-symlinks", action="store_true")
    ap.add_argument("--skip-unreadable", action="store_true")
    ap.add_argument("--allow-concurrent-writes", action="store_true")
    ap.add_argument("--escape-odd-names", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the plan, write nothing")
    ap.add_argument("--quiet", action="store_true",
                    help="with --verify: print only non-OK rows")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    patterns: list[SanitizationPattern] = []
    try:
        if args.patterns_file is not None:
            patterns.extend(load_patterns_file(args.patterns_file))
        if args.generic_patterns:
            patterns.extend(generic_patterns(contents=args.scan_bytes))
    except ManifestGeneratorError as exc:
        print(f"HALT: {exc}", file=sys.stderr)
        return 1

    if args.verify is not None:
        try:
            result = verify_manifest(args.verify, anchor=args.anchor,
                                     quiet=args.quiet)
        except ManifestGeneratorError as exc:
            print(f"HALT: {exc}", file=sys.stderr)
            return 1
        for finding in result.findings:
            print(f"FINDING: {finding}")
        print(f"{result.n_ok} OK · {len(result.failed)} FAILED · "
              f"{len(result.missing)} MISSING · {len(result.self_rows)} self-row(s)")
        return result.exit_code()

    if args.scan_only is not None:
        if not patterns:
            print("HALT: --scan-only needs --patterns-file and/or "
                  "--generic-patterns; an empty sweep reports a clean pass and "
                  "is the most convincing false negative available",
                  file=sys.stderr)
            return 1
        try:
            parsed = parse_manifest(args.scan_only)
        except ManifestGeneratorError as exc:
            print(f"HALT: {exc}", file=sys.stderr)
            return 1
        hits = scan_paths([row.name for row in parsed.rows], patterns,
                          show_matches=args.show_matches)
        for hit in hits:
            print(hit.describe())
        print(f"sanitization sweep over {len(parsed.rows)} row PATHS: "
              f"{len(hits)} hit(s)")
        #  The zero-assert IS the exit code (rake M46(a)).
        return 4 if hits else 0

    if args.output is None:
        ap.error("--tree requires --output")
    try:
        spec = ManifestSpec(
            tree=args.tree, output=args.output,
            anchor=args.anchor if args.anchor is not None else Path.cwd(),
            headers=list(args.header),
            exclude_globs=(list(args.exclude) if args.exclude is not None
                           else list(DEFAULT_EXCLUDE_GLOBS)),
            exclude_path_globs=list(args.exclude_path),
            sanitization_patterns=patterns,
            sanitize_mode=args.sanitize_mode,
            unsanitized_copy=args.unsanitized_copy,
            scan_bytes=args.scan_bytes, sort=args.sort,
            expect_rows=args.expect_rows,
            compare_previous=args.compare_previous,
            strict_broken_symlinks=args.strict_broken_symlinks,
            skip_unreadable=args.skip_unreadable,
            allow_concurrent_writes=args.allow_concurrent_writes,
            escape_odd_names=args.escape_odd_names, dry_run=args.dry_run)
    except ValueError as exc:
        print(f"HALT: {exc}", file=sys.stderr)
        return 1
    try:
        result = generate_tree_manifest(spec)
    except SanitizationLeakError as exc:
        print(f"HALT (rake M46): {exc}", file=sys.stderr)
        return 4
    except ManifestGeneratorError as exc:
        print(f"HALT: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"HALT (filesystem): {exc}", file=sys.stderr)
        return 1
    _print_build_report(result)
    #  Every condition that must block has already raised; findings are
    #  REPORTED, not converted into an exit code, so a legitimate broken symlink
    #  does not train an operator to ignore rc=1.
    return 0


if __name__ == "__main__":
    sys.exit(main())
