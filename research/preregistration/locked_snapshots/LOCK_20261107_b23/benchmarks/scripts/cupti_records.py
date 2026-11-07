"""Pure helpers for CUPTI activity records (name, start ns, duration ns), used by kernel_time_probe.py (B23).
No torch import, so they are unit-tested with plain python (test_cupti_records.py)."""

MARK_NAME = "_b23_marker"
NAME_LEN = 160


def after_marker(recs, mark=MARK_NAME):
    """Records after the last marker record (warm-up records before it are dropped); None if there is no marker."""
    idx = [i for i, r in enumerate(recs) if mark in r[0]]
    return recs[idx[-1] + 1:] if idx else None


def records_ok(recs, expect, reps):
    """True if recs are exactly `reps` repetitions of the expected kernel-name sequence `expect`."""
    if not expect or recs is None or len(recs) != reps * len(expect):
        return False
    return all(r[0][:NAME_LEN] == expect[i % len(expect)] for i, r in enumerate(recs))


def op_times(recs, ks):
    """Records of whole repetitions -> per op slot j (ks[j] kernels each) the list of op durations in us
    (summed over the op's kernels). Empty lists if ks has a non-positive entry or recs do not fill whole repetitions."""
    out = [[] for _ in ks]
    per = sum(ks)
    if not ks or min(ks) <= 0 or len(recs) % per:
        return out
    i = 0
    while i < len(recs):
        for j, k in enumerate(ks):
            out[j].append(sum(r[2] for r in recs[i:i + k]) / 1e3)
            i += k
    return out
