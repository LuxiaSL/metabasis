# Contributing

Two kinds of contribution arrive here, and they are handled differently.

**Code** is a pull request against this repository. A change merges when every touched
module that defines `--selftest` passes it, and a reviewer agrees the change keeps the rules
below. A touched module without a selftest needs a reviewer to run what it touches; adding
one is welcome.

**Claims** are never merged as claims. A contribution that asserts a finding — that a map
carries a vector, that a prediction lands, that a number holds — is raised as an issue here,
with the runs and receipts behind it: inputs by sha256, the command that produced the
number, and the output it printed. It is evaluated, replicated where warranted, and graded
before it enters [`RESULTS.md`](RESULTS.md). Code that *enables* a claim is welcome in a pull
request; the claim itself travels the other road.

## The documentation rule

Documentation and comments obey two rules at once. They apply to comments, docstrings, and
the message a `raise` or a logging call says out loud: a stranger meets that last one at the
moment something breaks, which is the worst moment to hand them a pointer they cannot follow.
They never apply to wire format — a dict key, a status value, a filename, a label banked
results carry — which is content the code keys on. They do apply to prose a run writes for a
human: a `note`, a `description` or a `reason` in an output is read the way a comment is read.

### 1. State what is true now

A comment describes the code as it stands. It does not narrate how the code got there.

No `previously`, `used to <verb>`, `changed in`, `we now`, `no longer`. No bare `TODO`,
`FIXME` or `HACK`: a known gap is a refusal the code makes explicitly, a test that pins the
current behaviour, or an issue.

**No dates.** Not on an edit, and not on evidence either. Git records when a line changed,
and a dated measurement belongs in the record that holds it. What the code needs is the
standing fact it depends on, in the present tense.

```python
# Good — the constraint, and why it is load-bearing.
# A proc fit with an isotropic scale is an isometry on span(va), so cosines between
# transported vectors equal cosines between their in-frame projections.

# Bad — narrates an edit.
# We now normalise by the site median norm; this used to fit raw states.

# Bad — dates the evidence instead of stating it.
# Rank guard k <= n_train / 1.2 (fixed in the second session).
```

### 2. Every referent must be reachable from this repository

If a comment or docstring names something, a reader holding only this repository must be
able to go look at it: a module path that exists here, a file in this tree, a symbol this
package defines, or a public URL.

It does **not** mean a path into a private tree, a planning note, an internal ticket, a
conversation, a machine, a host, an account, a scheduler, or a sibling checkout. Nor does it
mean a **provenance citation**: a `§` of a document that is not here, an arm or milestone
code, an addendum, a bare item code, a commit hash. The code is the receipt for what the code
does, and git is the receipt for how it came to do it. The two frozen pre-registrations in
[`docs/planning/`](docs/planning/) are in this tree and may be linked; a citation of a
section that only exists in a private record may not.

When the substance is short, state it inline. When it is long and public, link it. When it
is long and private, restate the part this code depends on: one sentence of standing fact
beats a citation nobody can open.

A consequence worth stating plainly: **a claim in a docstring must match the code under it.**

### Keep the prose style already here

Present tense. Say why, not only what — the constraint a reader would otherwise violate, the
failure a refusal exists to prevent. Document a refusal where the refusal happens. A
function's docstring carries what a caller needs: arguments, units, the shape of what comes
back, and how it fails.

## The gates

Each is a command you can run from the repository root. Install the test runner with
`uv pip install -e '.[dev]'`.

| gate | what it checks | command |
|---|---|---|
| tests | the suite, including the corpus that pins each documentation rule | `pytest tests/` |
| selftests | every touched module that defines one | `python -m metabasis.scripts.<module> --selftest` |
| state what is true now | no marker comments, no changelog phrasing, no dates in prose | `python -m tools.check_timelessness --root metabasis tools tests` |
| reachable referents | every path, module and document named in prose resolves inside this tree; no provenance citations; no prose bound to one operator's machines | `python -m tools.check_referents --root metabasis tools tests` |
| import closure | every module is reached from a script, a test, or a listed entry point; no import names a missing module | `python -m tools.check_import_closure --package metabasis --roots metabasis/scripts tests` |

All five pass on the current tree, and a pull request keeps them passing. Each checker
also takes `--report-only`, which prints the full receipt and exits 0.

Each documentation rule is pinned by a corpus rather than by reading:
`tests/test_gate_fixtures.py` holds the strings each checker must catch and the
legitimate prose it must stay quiet on. A rule that stops seeing something fails a test
there instead of quietly reporting a pass; closing a blind spot starts by adding the
string that slipped through.

The pattern files (`tools/timelessness_allowlist.txt`, `tools/referents_allowlist.txt`,
`tools/closure_allowlist.txt`) and `tools/frozen_modules.txt` carry a reason beside every
entry, and an entry is a claim that its reason is true now. The referent file's `[infra]`
section holds the shapes of prose bound to one operator's machines, never the names of
hosts, accounts or schedulers: a published list would defeat keeping them out, so a bare
name remains a reviewer's catch.

**Frozen modules.** The five modules of the steering instrument are listed in
`tools/frozen_modules.txt` and are not scanned. Every steering run records their sha256
as the instrument's identity, and two are held at fixed hashes, so any byte change,
documentation included, makes a different instrument. Their documentation is brought in
line only together with a recertification of the instrument.

**What the gates do not cover.** String values the code writes into artifacts or
compares against (stamp constants, status lines, recorded notes) are wire format and are
left as they are, even where they read like prose; changing them would make new records
disagree with banked ones. Selftest check labels are printed output, not documentation.

## Before you open a pull request

- Run the selftest of every touched module that has one:
  `python -m metabasis.scripts.<module> --selftest`. A selftest that skips names what
  evidence is missing; a new skip needs a reason. Not every module defines `--selftest`, and
  one that does not may treat the flag as unknown or ignore it, so check `--help` first.
- Nothing that identifies infrastructure enters the tree: no hostnames, usernames, absolute
  paths, credentials or keys. Values of that class come from the environment.
- `outputs/` never enters git. Data travels out-of-band and verifies against
  [`manifests/`](manifests/).
