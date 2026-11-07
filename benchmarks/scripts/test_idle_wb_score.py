"""Unit tests for idle_wb_score (CPU only; SYNTHETIC data)."""

import unittest

import idle_wb_score as s9

WS = (0.125, 0.25, 0.375)
CELLS = [(w, "host", t) for w in WS for t in (0.0, 0.1, 1.0, 10.0)] + [(w, "spin", 10.0) for w in WS]


def world(clean_fn, counted=True, extra=0.05, ctl_w=0.0, fails=0, a0=0.0):
    """clean_fn(w, idle, T) -> fraction cleaned during the idle period."""
    meta = {"correctness_failures": fails, "ranges": []}
    rows = []
    for _ in range(2):
        meta["ranges"].append({"kind": "control"})
        rows.append({"read": 9000.0, "write": ctl_w})
    for w, idle, t in CELLS:
        n = int(w * 3145728)
        for rep in range(3):
            c = clean_fn(w, idle, t)
            meta["ranges"].append({"kind": "A", "w": w, "idle": idle, "T_ms": t, "rep": rep, "bytes": n})
            rows.append({"read": 0.0, "write": n * ((c if counted else 0.0) + (a0 if t == 0 else 0.0))})
            meta["ranges"].append({"kind": "B", "w": w, "idle": idle, "T_ms": t, "rep": rep, "bytes": n})
            rows.append({"read": 16e6, "write": n * (1 - c - (a0 if t == 0 else 0.0)) + extra * 589824})
    return s9.score([meta], [rows], log=lambda *_: None)


class TestB9(unittest.TestCase):
    def test_idle_writeback(self):
        v = world(lambda w, idle, t: min(1.0, t / 2))
        self.assertTrue(v["I-0"])
        self.assertEqual(v["I-1"], "no write-back while writing below the cap")
        self.assertEqual(v["I-2"], "idle write-back exists")
        self.assertAlmostEqual(v["I-2 IWB"]["0.25"], 1.0, places=6)

    def test_no_idle_writeback(self):
        v = world(lambda w, idle, t: 0.0)
        self.assertTrue(v["I-0"])
        self.assertEqual(v["I-2"], "no idle write-back")

    def test_partial_and_not_counted(self):
        # range A does not see the idle write-back: IWB from B still finds it
        v = world(lambda w, idle, t: 0.9 if (t >= 10 and w < 0.3) else 0.0, counted=False)
        self.assertEqual(v["I-2"], "partial or size-dependent")
        self.assertAlmostEqual(v["I-2 IWB"]["0.125"], 0.9, places=6)

    def test_validity(self):
        self.assertFalse(world(lambda *_: 0.0, ctl_w=8000.0)["I-0 parts"]["b control writes"])
        self.assertFalse(world(lambda *_: 0.0, fails=1)["I-0 parts"]["a correctness"])
        self.assertFalse(world(lambda *_: 0.0, extra=0.3)["I-0 parts"]["c baseline A+B"])  # A+B = 1.45 at w 0.125

    def test_writeback_while_writing(self):
        self.assertEqual(world(lambda *_: 0.0, a0=0.2)["I-1"], "write-back while writing below the cap")


if __name__ == "__main__":
    unittest.main()
