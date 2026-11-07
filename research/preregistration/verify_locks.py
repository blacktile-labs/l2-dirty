#!/usr/bin/env python3
"""Check every pre-registration lock: each LOCK_*.sha256 lists sha256 hashes of the protocol, code and predictions
as they were when the lock was committed (before any measurement of that experiment was read). The files as they were
at that moment are in locked_snapshots/<LOCK>/; lock_provenance.json gives the commit and its date in the authors'
development history (self-attested timestamps).
  python3 research/preregistration/verify_locks.py
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    prov = json.loads((HERE / "lock_provenance.json").read_text())
    bad = total = 0
    for lock in sorted(HERE.glob("LOCK_*.sha256")):
        ok = n = 0
        for line in lock.read_text().splitlines():
            m = re.match(r"^([0-9a-f]{64})\s+\*?(.+)$", line.strip())
            if not m:
                continue
            n += 1
            f = HERE / "locked_snapshots" / lock.stem / m.group(2)
            ok += f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == m.group(1)
        total += n
        bad += n - ok
        p = prov.get(lock.name, {})
        print(f"{lock.name:32s} {ok}/{n} files match   committed {p.get('commit_date', '?')}  {p.get('commit', '?')[:12]}")
        if ok < n and p.get("note"):
            print("    note:", p["note"])
    print(f"\n{total - bad}/{total} locked files match their sha256")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
