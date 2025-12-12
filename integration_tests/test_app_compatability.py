from __future__ import annotations
import sys
import subprocess
from pathlib import Path

DATA = Path(__file__).parents[1] / "molecsim" / "tests" / "data"
APP_DIR = Path(__file__).parents[1] / "apps"


def run_cli(
    name: str, args: list[str], timeout: int = 15
) -> subprocess.CompletedProcess[str]:
    """Allow running of subprocess for apps."""
    return subprocess.run(
        [sys.executable, str(APP_DIR / name), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_showcase_smoke(tmp_path: Path):
    """Test if showcase app runs."""
    infile = DATA / "water_zmatrix.com"
    args = ["-f", str(infile), "-n", "2", "-t", "100", "-b", "10,10,10"]
    cp = run_cli("calculate_between.py", args, timeout=20)
    assert cp.returncode == 0
    assert "For a total of" in cp.stdout.lower() or cp.stdout
    assert cp.stderr == ""


def test_headless_smoke(tmp_path: Path):
    """Test if showcase app runs."""
    infile = DATA / "water_zmatrix.com"
    args = ["-f", str(infile), "-n", "2", "-t", "100", "-b", "10,10,10"]
    cp = run_cli("headless.py", args, timeout=20)
    assert cp.returncode == 0
    assert "For a total of" in cp.stdout.lower() or cp.stdout
    assert cp.stderr == ""


def test_debug_smoke(tmp_path: Path):
    """Test if showcase app runs."""
    infile = DATA / "water_zmatrix.com"
    args = ["-f", str(infile), "-n", "2", "-t", "100", "-b", "10,10,10"]
    cp = run_cli("show_traces.py", args, timeout=20)
    assert cp.returncode == 0
    assert cp.stderr == ""
