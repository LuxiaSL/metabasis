"""L4 micro-gold blind-judging tool (pre-statement §5-L4, brief 2026-08-07).

TOOLING ONLY. This package computes no science number and applies no
criterion. It draws pairs deterministically from the detection-PASS
population, presents them BLIND, and records Luxia's verdicts to an
append-only log plus a sealed sha'd artifact. Unblinding is a separate
desk step that joins the sealed map; nothing in here ever unblinds.

C§8: every artifact this package writes carries GRADE = "UNSTAMPED (C§8)".
"""
from __future__ import annotations

#: Tool version. Rides every artifact this package writes. Bump on ANY
#: change to the draw algebra (`draw.py`) or the verdict record shape.
TOOL_VERSION = "l4-blind-tool/1.0.0"

#: The C§8 grade string, verbatim, on everything written.
GRADE = "UNSTAMPED (C§8)"

#: The status banner every artifact carries — L4 gold is a human read, not
#: a quotable number, until the desk joins it and Luxia stamps it.
STATUS = (
    "L4 MICRO-GOLD (human verdicts). NOT QUOTABLE until the desk unblinds "
    "it against the sealed map and Luxia stamps the read. No verdict, no "
    "gate, no criterion comparison and no attribution is computed here."
)

__all__ = ["TOOL_VERSION", "GRADE", "STATUS"]
