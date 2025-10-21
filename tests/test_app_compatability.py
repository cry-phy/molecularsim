from __future__ import annotations
import sys
import subprocess
from pathlib import Path

DATA = Path(__file__).parent / "data"
APP = Path(__file__).parents[1] / "src" / "apps" / "calculate_between.py"


def run_cli(args: list[str], timeout: int = 15) -> subprocess.CompletedProcess[str]:
    """Allow running of subprocess for apps."""
    return subprocess.run(
        [sys.executable, str(APP), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_showcase_smoke(tmp_path: Path):
    """Test if showcase app runs."""
    infile = DATA / "water_zmatrix.com"
    args = ["-f", str(infile), "-n", "2", "-t", "100", "-b", "10,10,10"]
    cp = run_cli(args, timeout=20)
    assert cp.returncode == 0
    assert "For a total of" in cp.stdout.lower() or cp.stdout
    assert cp.stderr == ""
