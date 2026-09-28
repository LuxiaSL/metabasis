"""Command-line guard for scripts whose inputs and outputs are fixed paths.

Several readouts take no arguments: they read named files and write named files.
Without a parser, any argument — a mistyped flag, or `--selftest` on a script
that has none — would be silently ignored and the script would run its real job.
`parse_no_arguments` gives such a script a `--help` built from its docstring and
turns every other argument into a usage error (exit status 2) before anything
is read or written.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

EPILOG = ("This script takes no arguments: the paths it reads and writes are fixed "
          "and named above. Any argument other than --help is refused.")


def parse_no_arguments(doc: Optional[str],
                       argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Answer `--help` from `doc`; exit 2 on any other argument."""
    parser = argparse.ArgumentParser(
        description=(doc or "").strip() or None, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    return parser.parse_args(argv)


def selftest() -> int:
    """The guard refuses flags and positionals, and accepts an empty argv."""
    import contextlib
    import io

    failures: list[str] = []

    def refused(argv: list[str]) -> bool:
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parse_no_arguments("A fixed-path job.", argv)
            except SystemExit as exc:
                return exc.code == 2
        return False

    for argv in (["--selftest"], ["--no-such-flag"], ["positional"]):
        if not refused(argv):
            failures.append(f"{argv} was not refused")
    try:
        parse_no_arguments("A fixed-path job.", [])
    except SystemExit:
        failures.append("an empty argv was refused")
    for name in failures:
        print(f"  FAIL: {name}")
    print(f"_fixed_job selftest: {4 - len(failures)}/4 pass")
    return 1 if failures else 0


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ap.add_argument("--selftest", action="store_true", required=True,
                     help="check the guard itself")
    _ap.parse_args()
    raise SystemExit(selftest())
