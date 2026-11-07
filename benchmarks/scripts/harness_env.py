"""Environment / provenance capture shared by the benchmark runners (standard library only)."""

from __future__ import annotations

import datetime as dt
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sh(cmd: list[str], timeout: int = 30) -> dict:
    """Run a read-only query command; never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": " ".join(cmd), "rc": p.returncode, "out": p.stdout.strip(), "err": p.stderr.strip()}
    except FileNotFoundError:
        return {"cmd": " ".join(cmd), "rc": None, "out": "", "err": "not found"}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "rc": None, "out": "", "err": f"timeout after {timeout}s"}


def nvidia_smi_snapshot() -> dict:
    fields = ("name,driver_version,temperature.gpu,clocks.sm,clocks.mem,clocks.max.sm,"
              "clocks.max.mem,power.draw,pstate,memory.used,memory.total")
    snap = sh(["nvidia-smi", f"--query-gpu={fields}", "--format=csv"])
    # Throttle reasons were renamed in newer drivers; try both spellings.
    for f in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        r = sh(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader"])
        if r["rc"] == 0:
            snap["clock_limit_reasons"] = r["out"]
            break
    else:
        snap["clock_limit_reasons"] = "unavailable"
    return snap


def sha256_of(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def capture_env(binary: Path, sources: list[Path]) -> dict:
    build_dir = binary.parent

    def read(name: str) -> str:
        p = build_dir / name
        return p.read_text().strip() if p.exists() else "unavailable"

    return {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        # The toolchain that built the binary (written by build.sh), not whatever is on PATH.
        "build_nvcc_version": read("nvcc_version.txt"),
        "build_flags": read("build_flags.txt"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256_of(p) for p in sources},
        "binary_sha256": sha256_of(binary),
        "git_rev": sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": sh(["git", "-C", str(ROOT), "status", "--porcelain"]),
    }
