import unittest

import dobench_flush_score as s

L2 = 4 << 20


def run(xs, ctl_w=0.0, wr=0.0, dev="NVIDIA GeForce RTX 2060 SUPER", fails=0, t_read=100.0, dt=None):
    """xs: {w_read: X}; GEMV reads w*L2; 'read' writes wr*L2, 'zero' writes (wr + X)*L2."""
    ev, rows = [], []
    for _ in range(2):
        ev.append({"kind": "control"}); rows.append({"read": 0.0, "write": ctl_w})
    for w, x in xs.items():
        for rep in range(3):
            for var in ("zero", "read"):
                ev.append({"kind": "gemv", "variant": var, "w_read": w, "rep": rep, "weight_bytes": w * L2})
                rows.append({"read": w * L2, "write": (wr + (x if var == "zero" else 0)) * L2})
    rmeta = {"device": dev, "l2_bytes": L2, "events": ev, "correctness_failures": fails}
    tev = []
    for w, x in xs.items():
        rate = (w * L2 + wr * L2) / t_read
        d = (x * L2 / rate) if dt is None else dt
        tev += [{"kind": "timing", "w_read": w, "variant": "read", "times_us": [t_read] * 5},
                {"kind": "timing", "w_read": w, "variant": "zero", "times_us": [t_read + d] * 5}]
    tmeta = {"events": tev, "correctness_failures": 0}
    return [rmeta], list(zip(ev, rows)), [tmeta]


class DoBenchFlushScore(unittest.TestCase):
    def sc(self, *a, **k):
        return s.score(*run(*a, **k), log=lambda *_: None)

    def test_matches_b9b_residue_and_bytes_explain_time(self):
        v = self.sc({2: 0.50, 4: 0.51, 8: 0.49})
        self.assertTrue(v["G-0"])
        self.assertEqual(v["G-1"], "residue = B9b-predicted dirty residue")
        self.assertEqual(v["G-1b"], "residue independent of the GEMV size")
        self.assertTrue(v["G-2"].startswith("timing bias = residue"))

    def test_no_residue(self):
        v = self.sc({2: 0.0, 4: 0.01, 8: 0.0}, dt=0.5)
        self.assertEqual(v["G-1"], "not")

    def test_ada_prediction_discriminates_b9b_from_v4(self):
        v = self.sc({2: 0.68, 4: 0.70, 8: 0.66}, dev="NVIDIA GeForce RTX 4080 SUPER")
        self.assertEqual(v["G-1"], "residue = B9b-predicted dirty residue")
        self.assertEqual(v["G-1 closer to"], "B9b residue")
        v = self.sc({2: 0.50, 4: 0.50, 8: 0.50}, dev="NVIDIA GeForce RTX 4080 SUPER")
        self.assertEqual(v["G-1 closer to"], "v4 (0.5)")

    def test_time_not_explained(self):
        v = self.sc({2: 0.5, 4: 0.5, 8: 0.5}, dt=100.0)
        self.assertEqual(v["G-2"], "not explained by bytes alone")

    def test_sanity_failures(self):
        self.assertFalse(self.sc({2: 0.5, 4: 0.5, 8: 0.5}, ctl_w=1 << 20)["G-0"])
        self.assertFalse(self.sc({2: 0.5, 4: 0.5, 8: 0.5}, wr=0.2)["G-0"])
        self.assertFalse(self.sc({2: 0.5, 4: 0.5, 8: 0.5}, fails=1)["G-0"])

    def test_range_rows_count_check(self):
        rm, rr, _ = run({2: 0.5})
        with self.assertRaises(ValueError):
            s.range_rows(rm, [[r for _, r in rr][:-1]])


if __name__ == "__main__":
    unittest.main()
