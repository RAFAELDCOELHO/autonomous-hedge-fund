"""`make reproduce` end to end: run the Makefile recipe itself, byte-identical.

tests/test_reproduce.py pins each paper artifact by calling the scripts' mains
in-process with outputs redirected to a temp dir. This test closes the gap to
the Makefile: it parses the `reproduce:` recipe, runs each command (``$(PY)``
-> this interpreter) in a scratch copy of the repo, offline (yfinance poisoned
in every child process), and requires every file under benchmark/results/ and
docs/ to come out byte-identical to the committed tree, with no new or missing
files. It also records that the pipeline never imports
tradingagents.backtest.runner, so runner/baseline changes cannot move it.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COPY = ("scripts", "tradingagents", "benchmark", "docs", "data", "Makefile")
COMPARED = ("benchmark/results", "docs")

SITECUSTOMIZE = """
import atexit, os, sys
sys.modules["yfinance"] = None  # any download attempt raises
def _record():
    with open(os.environ["REPRO_MARKER"], "a") as fh:
        fh.write(("runner" if "tradingagents.backtest.runner" in sys.modules else "clean") + "\\n")
atexit.register(_record)
"""


def _recipe() -> list[list[str]]:
    lines = (REPO / "Makefile").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("reproduce:"))
    cmds = []
    for line in lines[start + 1:]:
        if not line.startswith("\t"):
            break
        cmds.append(shlex.split(line.strip().replace("$(PY)", "PYTHON")))
    return cmds


def _files(root: Path) -> dict[str, bytes]:
    out = {}
    for sub in COMPARED:
        for path in sorted((root / sub).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                out[path.relative_to(root).as_posix()] = path.read_bytes()
    return out


def test_make_reproduce_recipe_is_the_pipeline_test_reproduce_pins():
    assert _recipe() == [
        ["PYTHON", "scripts/run_random_n100.py"],
        ["PYTHON", "scripts/run_ew_portfolio.py"],
        ["PYTHON", "scripts/run_brazilbench.py", "--write"],
    ]


def test_make_reproduce_output_is_byte_identical(tmp_path):
    work = tmp_path / "repo"
    work.mkdir()
    for name in COPY:
        src = REPO / name
        if src.is_dir():
            shutil.copytree(src, work / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, work / name)
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "sitecustomize.py").write_text(SITECUSTOMIZE)
    marker = tmp_path / "marker.txt"
    env = {**os.environ, "PYTHONPATH": f"{poison}{os.pathsep}{work}", "REPRO_MARKER": str(marker),
           "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("ANTHROPIC_API_KEY", None)
    before = _files(work)
    for path in (work / "benchmark/results").rglob("*"):  # so a script that writes elsewhere fails
        if path.is_file():
            os.utime(path, (0, 0))

    for cmd in _recipe():
        cmd = [sys.executable if part == "PYTHON" else part for part in cmd]
        run = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=600)
        assert run.returncode == 0, (cmd, run.stdout[-2000:], run.stderr[-2000:])

    after = _files(work)
    rewritten = {
        "benchmark/results/random_n100/summary.csv", "benchmark/results/random_n100/per_seed_returns.csv",
        "benchmark/results/baselines/per_cell_returns.csv", "benchmark/results/baselines/ew_returns.csv",
        "benchmark/results/brazilbench/matrix.csv",
    }
    for name in rewritten:
        assert (work / name).stat().st_mtime > 0, f"{name} was not regenerated in the scratch copy"
    assert sorted(after) == sorted(before), set(after) ^ set(before)
    drifted = [name for name in before if after[name] != before[name]]
    assert drifted == [], f"make reproduce drifted: {drifted}"
    assert after == _files(REPO)  # the copy started from the committed working tree
    assert marker.read_text().split() == ["clean", "clean", "clean"]  # 3 commands, runner never imported
