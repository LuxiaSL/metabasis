"""Checkpoint-identity discriminators, valued on BOTH sides, and the check that
they actually separate — rake M17(a) as code.

WHY THIS EXISTS. Rake M9 says checkpoint identity is judged by metadata, never
by a directory name. Rake M17 says the metadata SET is per-lineage and must be
SHOWN to work before it is trusted: on the Mixtral lineage the base and instruct
repos ship a byte-identical `config.json` AND a byte-identical
`generation_config.json`, so the reflexive Llama/Qwen recipe passes silently on
the wrong checkpoint; on the DeepSeek-V3 lineage the ENTIRE metadata surface
degenerates and only the weights separate; while on the Qwen3 lineage the
chat template — load-bearing for Mixtral — proves nothing at all, because Qwen
ships templates on base checkpoints too.

Prose in a registry note cannot be run. This module records, per lineage, the
metadata fields with their values ON BOTH SIDES, and computes which of them
actually separate the confusable pair. `separation_problems()` then asserts that
every field a roster node RELIES on is one that demonstrably separates, and
`unregistered_lineages()` names the roster nodes for which no separation demo
exists yet, so the gap is a listed row rather than an unexamined assumption.

Everything here is OFFLINE: the values are digests already recorded in
`metabasis.roster` node notes and in the desk's verification reports. Nothing
fetches, nothing loads weights, nothing needs a checkpoint on disk. That is the
point — the demo has to be re-runnable long after the weights move.

READING A DIGEST PAIR. Some digests are on record only as leading hex (the way
the verification reports quote them). A pair is compared over the COMMON prefix:
differing there PROVES separation; agreeing there with unequal recorded lengths
proves nothing and is reported as INCONCLUSIVE rather than counted as a pass.

Run (repo root, PYTHONPATH=.):
  python -m metabasis.lineage_discriminators
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from metabasis.roster import ROSTER

#: `separates` — the two sides differ where both are recorded, so the field
#: identifies WHICH checkpoint this is. `degenerate` — the two sides are
#: identical (or identically absent): the field proves lineage membership at
#: most, never membership of a particular repo. `inconclusive` — the recorded
#: values agree as far as both were recorded, but one is a shorter prefix, so
#: nothing is proved either way.
Separation = Literal["separates", "degenerate", "inconclusive"]


class MetadataField(BaseModel):
    """One identity field, valued on the checkpoint of record AND its sibling.

    `None` means the artifact is ABSENT on that side, which is itself a value:
    the Mixtral base repo ships no chat template at all, and that absence is
    what makes template presence load-bearing on exactly that lineage.
    """
    field: str = Field(description="the metadata artifact, named as it is on disk "
                                   "(config.json, tokenizer_config.json, …) or as "
                                   "the property it reads (chat_template, eos_ids)")
    of_record: Optional[str] = Field(
        description="value on the checkpoint of record; None => artifact ABSENT")
    sibling: Optional[str] = Field(
        description="value on the confusable sibling; None => artifact ABSENT")
    note: str = ""

    @property
    def separation(self) -> Separation:
        a, b = self.of_record, self.sibling
        if a is None and b is None:
            return "degenerate"
        if a is None or b is None:
            return "separates"          # presence vs absence IS a separation
        n = min(len(a), len(b))
        if a[:n] != b[:n]:
            return "separates"
        return "degenerate" if len(a) == len(b) else "inconclusive"

    @property
    def separates(self) -> bool:
        return self.separation == "separates"


class LineageIdentity(BaseModel):
    """One roster node, its confusable sibling, and the fields that tell them apart.

    `discriminators_of_record` is the CLAIM — the fields the campaign relies on
    for this node. Every one of them must come out `separates`, and that is what
    `separation_problems()` checks; a field that merely happens to separate is
    reported as corroboration, never as the claim.
    """
    model_key: str = Field(description="roster bank key (must exist in ROSTER)")
    checkpoint_of_record: str = Field(description="hub repo id of the checkpoint "
                                                  "the campaign collected")
    sibling: str = Field(description="hub repo id of the confusable sibling")
    why_confusable: str
    fields: tuple[MetadataField, ...]
    discriminators_of_record: tuple[str, ...]

    @model_validator(mode="after")
    def _keys_resolve(self) -> "LineageIdentity":
        if self.model_key not in ROSTER:
            raise ValueError(
                f"{self.model_key!r} is not a roster key {sorted(ROSTER)} — an "
                f"identity demo for a node that does not exist proves nothing")
        named = {f.field for f in self.fields}
        unknown = [d for d in self.discriminators_of_record if d not in named]
        if unknown:
            raise ValueError(
                f"{self.model_key}: discriminator(s) {unknown} carry no valued "
                f"field here, so they cannot be shown to separate anything "
                f"(valued fields: {sorted(named)})")
        if not self.discriminators_of_record:
            raise ValueError(
                f"{self.model_key}: no discriminator of record — an entry that "
                f"claims nothing cannot be checked")
        return self

    @property
    def separating(self) -> tuple[str, ...]:
        return tuple(f.field for f in self.fields if f.separates)

    @property
    def degenerate(self) -> tuple[str, ...]:
        return tuple(f.field for f in self.fields
                     if f.separation == "degenerate")


# ────────────────────────────────────────────────────────────────────────────
# The registry. Every value below is a digest already on record in the roster
# node notes / the desk's verification reports; nothing here was re-derived.
# ────────────────────────────────────────────────────────────────────────────
LINEAGES: tuple[LineageIdentity, ...] = (
    LineageIdentity(
        model_key="mixtral-8x7b-instruct-v0.1",
        checkpoint_of_record="mistralai/Mixtral-8x7B-Instruct-v0.1",
        sibling="mistralai/Mixtral-8x7B-v0.1",
        why_confusable=(
            "THE RAKE-M17 ORIGIN CASE. Base and instruct ship a BYTE-IDENTICAL "
            "config.json and a byte-identical generation_config (a bare "
            "`_from_model_config` stub: eos 2 / bos 1, no sampling defaults, so "
            "there is no temperature/top_p signature either). Even the shard "
            "sizes agree. The whole reflexive Llama/Qwen discriminator set "
            "degenerates here and would pass on the wrong checkpoint."),
        fields=(
            MetadataField(
                field="config.json",
                of_record="9d56d04b36d0fd12ff54ae4c5bac769cc176e254e64ff71144614b6318b40793",
                sibling="9d56d04b36d0fd12ff54ae4c5bac769cc176e254e64ff71144614b6318b40793",
                note="byte-identical — the usual first discriminator, useless here"),
            MetadataField(
                field="generation_config.json", of_record="40e6ecbc",
                sibling="40e6ecbc",
                note="byte-identical stub on both sides; no sampling signature"),
            MetadataField(
                field="tokenizer_config.json", of_record="47536143",
                sibling="747ec9",
                note="THE VALUE OF RECORD for this lineage (owed by the "
                     "big-model scans and recorded here): 47536143… is the "
                     "instruct repo's tokenizer_config, 747ec9… the base "
                     "repo's. Recorded as leading hex, and they differ inside "
                     "the common prefix, so the separation is proved."),
            MetadataField(
                field="chat_template",
                of_record="796853172235ab3aaa4810013ddfe2eab481bda3306dad4f700c00d436eed596",
                sibling=None,
                note="the 1058-char v0.1 [INST] template, carried inside "
                     "tokenizer_config; the base repo carries NONE. The one "
                     "rung where template PRESENCE is load-bearing — the exact "
                     "inverse of the Qwen3 entry below. Full digest reproduced "
                     "from the collection stamp of the banked run."),
            MetadataField(
                field="first_shard.sha256", of_record="54669c5a",
                sibling="b43400ce",
                note="rake M17(b)'s universal fallback: the weights always "
                     "separate. The checkpoint of record matches the former."),
        ),
        discriminators_of_record=("tokenizer_config.json", "chat_template",
                                  "first_shard.sha256")),
    LineageIdentity(
        model_key="dsv3",
        checkpoint_of_record="deepseek-ai/DeepSeek-V3",
        sibling="deepseek-ai/DeepSeek-V3-Base",
        why_confusable=(
            "M17 IN ITS PUREST FORM: the entire metadata surface degenerates. "
            "Identical config.json, identical tokenizer_config.json, hence an "
            "identical chat template — so template presence is not merely "
            "uninformative here but actively misleading — and NEITHER repo "
            "ships a generation_config.json, so there is no eos/sampling "
            "signature to fall back on. Only the weights separate."),
        fields=(
            MetadataField(
                field="config.json",
                of_record="cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9",
                sibling="cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9",
                note="BYTE-IDENTICAL is the recorded row-21 finding, so the "
                     "sibling carries the same full digest — recorded at full "
                     "length on both sides deliberately, because an 8-hex "
                     "sibling value would read INCONCLUSIVE here and understate "
                     "how completely this field degenerates"),
            MetadataField(field="tokenizer_config.json", of_record="637bcd1a",
                          sibling="637bcd1a", note="byte-identical"),
            MetadataField(field="chat_template", of_record="3b8267e5",
                          sibling="3b8267e5",
                          note="identical template on BOTH — presence and sha "
                               "are both misleading on this lineage"),
            MetadataField(field="generation_config.json", of_record=None,
                          sibling=None,
                          note="absent on both sides; nothing to compare"),
            MetadataField(field="first_shard.sha256", of_record="b933b099",
                          sibling="3f4e5fce",
                          note="the discriminator of record — matches the hub "
                               "LFS digest of deepseek-ai/DeepSeek-V3"),
        ),
        discriminators_of_record=("first_shard.sha256",)),
    LineageIdentity(
        model_key="dsv3",
        checkpoint_of_record="deepseek-ai/DeepSeek-V3",
        sibling="deepseek-ai/DeepSeek-V3.1",
        why_confusable=(
            "The other way to collect the wrong DeepSeek: a later revision of "
            "the same chat lineage. Refuted on config alone — and it MUST be, "
            "because its quantization_config carries scale_fmt 'ue8m0', which "
            "the pinned transformers does not know, so the block scales would "
            "be mis-read rather than rejected."),
        fields=(
            MetadataField(
                field="config.json",
                of_record="cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9",
                sibling="3e5d192d", note="differs in the first byte pair"),
        ),
        discriminators_of_record=("config.json",)),
    LineageIdentity(
        model_key="qwen3-30b-a3b",
        checkpoint_of_record="Qwen/Qwen3-30B-A3B",
        sibling="Qwen/Qwen3-30B-A3B-Base",
        why_confusable=(
            "Same nameplate, adjacent directories on a shared weight store, and "
            "the row asks for native+raw — i.e. a chat model — so collecting the "
            "Base sibling would silently fail the arm-consistency rule. THE "
            "INVERSE OF THE MIXTRAL CASE: here config and eos separate cleanly "
            "while template presence proves nothing, because Qwen publishes "
            "chat templates on Base checkpoints too (rake M9)."),
        fields=(
            MetadataField(
                field="config.json",
                of_record="2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297",
                sibling="7e414215", note="different object, first byte pair on"),
            MetadataField(field="eos_token_id", of_record="151645",
                          sibling="151643",
                          note="<|im_end|> (chat) vs <|endoftext|> (base)"),
            MetadataField(field="chat_template_presence", of_record="present",
                          sibling="present",
                          note="DEGENERATE BY DESIGN, and recorded so nobody "
                               "reaches for it: Qwen ships templates on Base "
                               "checkpoints. This row is the M17 lesson stated "
                               "in data rather than in prose."),
        ),
        discriminators_of_record=("config.json", "eos_token_id")),
    LineageIdentity(
        model_key="llama-3.3-70b-instruct",
        checkpoint_of_record="meta-llama/Llama-3.3-70B-Instruct",
        sibling="meta-llama/Llama-3.1-70B-Instruct",
        why_confusable=(
            "Not a base/instruct pair — a VINTAGE pair, and the more dangerous "
            "kind here because row 12 exists precisely to be compared against "
            "row 11. The two checkpoints are architecturally identical (80 "
            "layers, hidden 8192, same rope_scaling, same vocab), carry the "
            "SAME chat-template sha as each other and as the banked 8B hub, and "
            "carry the same eos ids and the same sampling defaults. Everything "
            "except config.json degenerates."),
        fields=(
            MetadataField(
                field="config.json",
                of_record="95ef9768e4741543dbfaf0c274f101855883ff338b235c99eca2b6a4f4abee12",
                sibling="fa6e9124e4621df77aecf96fbfaf7975814013d2d5ab1c972e965000588a9749",
                note="both byte-identical to their own hub repos; different "
                     "from each other"),
            MetadataField(field="chat_template", of_record="e10ca381",
                          sibling="e10ca381",
                          note="identical to each other AND to the banked 8B "
                               "hub — which is the POINT for the native arm "
                               "(a difference cannot be a templating "
                               "difference), and useless for identity"),
            MetadataField(field="eos_token_id", of_record="128001,128008,128009",
                          sibling="128001,128008,128009",
                          note="the Llama-3 instruct signature on both"),
            MetadataField(field="generation_config.sampling",
                          of_record="temperature=0.6,top_p=0.9",
                          sibling="temperature=0.6,top_p=0.9",
                          note="identical instruct sampling defaults"),
        ),
        discriminators_of_record=("config.json",)),
)


def separation_problems(lineages: tuple[LineageIdentity, ...] = LINEAGES
                        ) -> list[str]:
    """Every recorded discriminator that does NOT demonstrably separate.

    This is rake M17(a) mechanized: "before trusting any identity check, fetch
    the confusable siblings' metadata and show the chosen discriminator actually
    separates them." An empty list means every claim in the registry is backed
    by two recorded values that differ.
    """
    problems: list[str] = []
    for entry in lineages:
        by_name = {f.field: f for f in entry.fields}
        for name in entry.discriminators_of_record:
            field = by_name[name]
            if field.separation == "separates":
                continue
            problems.append(
                f"{entry.model_key} vs {entry.sibling}: {name} is claimed as a "
                f"discriminator of record but is {field.separation.upper()} — "
                f"of_record={field.of_record!r}, sibling={field.sibling!r}"
                + ("; the recorded values agree over every character both sides "
                   "recorded, so a longer digest is needed before this field "
                   "may be relied on"
                   if field.separation == "inconclusive" else
                   "; the two checkpoints are indistinguishable on this field"))
        if not entry.separating:
            problems.append(
                f"{entry.model_key} vs {entry.sibling}: NO recorded field "
                f"separates the pair at all — identity rests on nothing")
    return problems


def unregistered_lineages(lineages: tuple[LineageIdentity, ...] = LINEAGES
                          ) -> list[str]:
    """Roster nodes with no sibling-separation demo — the gap, named as rows.

    Absence is a RESULT, not a failure: several roster nodes have no confusable
    sibling worth the name, and for others (the 405B) the sibling's metadata was
    never recorded on our side. Either way it belongs on a list rather than in
    nobody's head.
    """
    covered = {entry.model_key for entry in lineages}
    return [f"{key} ({node.model_id}, roster row {node.roster_row}): no "
            f"sibling-separation demo on record"
            for key, node in sorted(ROSTER.items()) if key not in covered]


def table(lineages: tuple[LineageIdentity, ...] = LINEAGES) -> str:
    """The demo, as a table: per lineage, which fields separate and which do not."""
    lines: list[str] = []
    for entry in lineages:
        lines.append(f"\n{entry.model_key}  —  {entry.checkpoint_of_record}")
        lines.append(f"  confusable sibling: {entry.sibling}")
        width = max(len(f.field) for f in entry.fields)
        for field in entry.fields:
            mark = {"separates": "SEPARATES", "degenerate": "degenerate",
                    "inconclusive": "INCONCLUSIVE"}[field.separation]
            claimed = " *of record*" if field.field in entry.discriminators_of_record else ""
            lines.append(
                f"    {field.field:<{width}}  {mark:<12}  "
                f"{(field.of_record or '(absent)'):<20.20} vs "
                f"{(field.sibling or '(absent)'):<20.20}{claimed}")
    return "\n".join(lines)


def _comparator_selftest() -> list[str]:
    """Prove the comparator can FAIL, so a green run is not a rubber stamp.

    Rake M19(c) in miniature: a check whose failing branch is never exercised
    proves only that a process exited 0. Every separation verdict is driven here
    on synthetic values, including the case the registry must never contain.
    """
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    cases: tuple[tuple[Optional[str], Optional[str], Separation, str], ...] = (
        ("abcd1234", "abcd9999", "separates", "digests differing inside the prefix"),
        ("abcd1234", "abcd1234", "degenerate", "identical digests of equal length"),
        ("abcd1234ef", "abcd1234", "inconclusive",
         "agreeing over the common prefix with unequal recorded lengths"),
        ("abcd1234", None, "separates", "present vs ABSENT"),
        (None, None, "degenerate", "absent on both sides"),
    )
    for of_record, sibling, want, label in cases:
        got = MetadataField(field="probe", of_record=of_record,
                            sibling=sibling).separation
        check(got == want, f"{label} → {got} (want {want})")

    #  The case the registry must never hold: a claim resting on a field that
    #  does not separate. The checker has to say so.
    rubber_stamp = LineageIdentity(
        model_key=sorted(ROSTER)[0], checkpoint_of_record="of/record",
        sibling="of/sibling", why_confusable="synthetic",
        fields=(MetadataField(field="config.json", of_record="same",
                              sibling="same"),
                MetadataField(field="first_shard.sha256", of_record="aa",
                              sibling="bb")),
        discriminators_of_record=("config.json",))
    problems = separation_problems((rubber_stamp,))
    check(len(problems) == 1 and "DEGENERATE" in problems[0],
          f"a claim resting on a degenerate field FAILS: {problems[0][:80]!r}")
    prefix_only = LineageIdentity(
        model_key=sorted(ROSTER)[0], checkpoint_of_record="of/record",
        sibling="of/sibling", why_confusable="synthetic",
        fields=(MetadataField(field="config.json", of_record="abcdef01",
                              sibling="abcd"),),
        discriminators_of_record=("config.json",))
    problems = separation_problems((prefix_only,))
    check(any("INCONCLUSIVE" in p for p in problems)
          and any("NO recorded field separates" in p for p in problems),
          "a claim resting on a prefix that only AGREES so far FAILS twice — "
          "the claim is unproved AND the entry separates nothing at all "
          "(never passed on the strength of a longer digest nobody recorded)")
    for bad_key, label in (("no-such-model", "an unknown roster key"),):
        try:
            LineageIdentity(model_key=bad_key, checkpoint_of_record="a",
                            sibling="b", why_confusable="synthetic",
                            fields=(MetadataField(field="config.json",
                                                  of_record="a", sibling="b"),),
                            discriminators_of_record=("config.json",))
            check(False, f"{label} must be refused")
        except ValueError:
            check(True, f"{label} is refused at construction")
    try:
        LineageIdentity(model_key=sorted(ROSTER)[0], checkpoint_of_record="a",
                        sibling="b", why_confusable="synthetic",
                        fields=(MetadataField(field="config.json",
                                              of_record="a", sibling="b"),),
                        discriminators_of_record=("tokenizer_config.json",))
        check(False, "a discriminator with no valued field must be refused")
    except ValueError:
        check(True, "a discriminator naming no valued field is refused")
    return failures


def main() -> int:
    """Print the demo and exit NONZERO if any claimed discriminator fails it."""
    print("SIBLING-SEPARATION DEMO — rake M17(a): every discriminator of "
          "record, shown to separate its lineage's confusable sibling")
    print("\n== comparator selftest: the check can FAIL ==")
    comparator_failures = _comparator_selftest()
    print(table())
    problems = separation_problems() + [
        f"comparator selftest: {msg}" for msg in comparator_failures]
    gaps = unregistered_lineages()
    print(f"\n{len(LINEAGES)} lineage demo(s) over "
          f"{len({e.model_key for e in LINEAGES})} roster node(s)")
    for gap in gaps:
        print(f"  NAMED GAP: {gap}")
    for problem in problems:
        print(f"  FAILURE: {problem}")
    print(f"sibling separation: {'PASSED' if not problems else 'FAILED'} "
          f"({len(gaps)} roster node(s) with no demo on record)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
