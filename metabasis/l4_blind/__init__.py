"""L4 micro-gold blind-judging tool: human pairwise verdicts on steered text.

TOOLING ONLY. This package computes no science number and applies no
criterion. It draws pairs deterministically from the detection-PASS
population, presents them BLIND, and records the judge's verdicts to an
append-only log plus a sealed sha'd artifact. Unblinding is a separate
step that joins the sealed map; nothing in here ever unblinds.

Every artifact this package writes carries the `GRADE` string below, which
marks it as not yet stamped for quotation.
"""
from __future__ import annotations

#: Tool version. Rides every artifact this package writes. Bump on ANY
#: change to the draw algebra (`draw.py`) or the verdict record shape.
TOOL_VERSION = "l4-blind-tool/1.0.0"

#: The grade string, verbatim, on everything written. It is wire format: the
#: sealed artifacts carry it and a reader keys on it, so it never changes.
GRADE = "UNSTAMPED (C§8)"

#: The status banner every artifact carries — L4 gold is a human read, not
#: a quotable number, until it is unblinded against the sealed map and the
#: read is stamped. Wire format, like `GRADE`: sealed artifacts carry it verbatim.
STATUS = (
    "L4 MICRO-GOLD (human verdicts). NOT QUOTABLE until the desk unblinds "
    "it against the sealed map and Luxia stamps the read. No verdict, no "
    "gate, no criterion comparison and no attribution is computed here."
)

__all__ = ["TOOL_VERSION", "GRADE", "STATUS"]
