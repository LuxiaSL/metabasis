"""The gate machinery this repository adds around the ported checkers.

Three behaviours are pinned here, each on a throwaway tree so the result does not
depend on the state of the package: the import closure's entry-point file (a
listed module seeds the traversal, a line without a reason is refused, a stale
name fails), the referent index reading what git would publish rather than
whatever sits on disk, and `--report-only` changing the exit status and nothing
else.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools import check_import_closure, check_referents, check_timelessness


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def package(tmp_path: Path) -> Path:
    """A package with one script, one module only a `python -m` entry point imports."""
    root = tmp_path / "pkg"
    _write(root / "__init__.py", "")
    _write(root / "scripts" / "__init__.py", "")
    _write(root / "scripts" / "run.py", "import pkg.used\n")
    _write(root / "used.py", "VALUE = 1\n")
    _write(root / "entry.py", "import pkg.helper\n")
    _write(root / "helper.py", "VALUE = 2\n")
    return root


def test_an_unlisted_entry_point_and_what_it_imports_are_orphans(package: Path) -> None:
    report = check_import_closure.build_report(package, [package / "scripts"])
    assert report.orphans == ["pkg.entry", "pkg.helper"]
    assert not report.passed


def test_a_listed_entry_point_seeds_the_traversal(package: Path) -> None:
    report = check_import_closure.build_report(
        package, [package / "scripts"], entry_points=["pkg.entry"]
    )
    assert report.orphans == []
    assert report.entry_points == ["pkg.entry"]
    assert report.passed


def test_an_entry_point_naming_no_module_fails_and_a_reached_one_is_redundant(package: Path) -> None:
    report = check_import_closure.build_report(
        package, [package / "scripts"], entry_points=["pkg.entry", "pkg.used", "pkg.gone"]
    )
    assert report.unknown_entry_points == ["pkg.gone"]
    assert report.redundant_entry_points == ["pkg.used"]
    assert not report.passed


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("pkg.entry\n", "carries no reason"),
        ("pkg.entry   #   \n", "carries no reason"),
        ("not a module # reason\n", "not a dotted module name"),
        ("pkg.entry # one\npkg.entry # two\n", "listed twice"),
    ],
)
def test_the_entry_point_file_refuses_what_it_cannot_vouch_for(
    tmp_path: Path, line: str, message: str
) -> None:
    listing = tmp_path / "entries.txt"
    listing.write_text("# header\n\n" + line, encoding="utf-8")
    with pytest.raises(check_import_closure.ClosureError, match=message):
        check_import_closure.load_allowlist(listing)


def test_the_shipped_entry_point_file_parses_and_every_entry_has_a_reason() -> None:
    entries = check_import_closure.load_allowlist(check_import_closure.DEFAULT_ALLOWLIST)
    assert entries
    assert all(reason for reason in entries.values())


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_the_referent_index_is_what_git_would_publish(tmp_path: Path) -> None:
    """A gitignored directory on disk is not something a reader of the repository can open."""
    repo = tmp_path / "repo"
    _write(repo / ".gitignore", "private/\n")
    _write(repo / "pkg" / "__init__.py", "")
    _write(repo / "pkg" / "kept.py", "VALUE = 1\n")
    _write(repo / "private" / "notes.md", "not published\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    index = check_referents.TreeIndex.build(repo)
    assert index.contains("pkg/kept.py")
    assert not index.contains("private/notes.md")
    assert "private" not in index.top_level


def test_outside_a_work_tree_the_index_walks_the_directory(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    _write(tree / "pkg" / "__init__.py", "")
    _write(tree / "notes.md", "text\n")
    index = check_referents.TreeIndex.build(tree)
    assert index.contains("notes.md")
    assert "pkg" in index.top_level


def test_report_only_changes_the_exit_status_and_nothing_else(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = tmp_path / "dated.py"
    module.write_text("# measured 2026-07-12\nVALUE = 1\n", encoding="utf-8")
    assert check_timelessness.main(["--root", str(module)]) == 1
    strict = capsys.readouterr().out
    assert check_timelessness.main(["--root", str(module), "--report-only"]) == 0
    assert capsys.readouterr().out == strict
    assert "dated-prose: 1" in strict
