"""Unit tests for queued_wb_score (CPU only; SYNTHETIC data)."""

import unittest

import queued_wb_score as q

WS = (0.375, 0.75, 1.0, 1.5)
PRED = {w: {"A": max(0.0, w - 0.5) / w, "B": min(w, 0.5) / w} for w in WS}


def world(loss_fn, a_fn=None, w_fn=None, ctl=0.0, fails=0):
    """loss_fn(w): fraction lost by per-kernel mode; a_fn/w_fn: measured A / W (default v4)."""
    a_fn = a_fn or (lambda w: PRED[w]["A"])
    w_fn = w_fn or (lambda w: a_fn(w) - 0.0)
    kpairs, rr = [], [({"kind": "control"}, {"write": ctl})]
    for w in WS:
        for _ in range(6):
            kpairs.append({"w": w, "W": w_fn(w), "F": 1.02 - w_fn(w) - loss_fn(w)})
            e = {"w": w, "bytes": 1}
            rr += [(dict(e, kind="A"), {"write": a_fn(w)}), (dict(e, kind="B"), {"write": 1.02 - a_fn(w)}),
                   (dict(e, kind="C"), {"write": 1.02})]
    return q.score(kpairs, rr, fails, PRED, log=lambda *_: None)


class TestB9b(unittest.TestCase):
    def test_v4_prediction(self):
        p = q.v4_predict(0.75, 3145728, 16 << 20)
        self.assertAlmostEqual(p["A"], 0.25 / 0.75, places=3)
        self.assertAlmostEqual(p["A"] + p["B"], 1.0, places=6)
        self.assertEqual(q.v4_predict(0.375, 3145728, 16 << 20)["A"], 0.0)

    def test_loss(self):
        v = world(lambda w: 0.0 if w <= 0.375 else 0.1)
        self.assertTrue(v["Q-0"])
        self.assertEqual(v["Q-1"], "per-kernel mode loses write-backs above the cap")
        self.assertEqual(v["Q-3"], "v4 predicts one write kernel's write-backs")

    def test_no_loss(self):
        v = world(lambda w: 0.0)
        self.assertEqual(v["Q-1"], "no loss")
        self.assertEqual(v["Q-2"], "the kernel window counts them")

    def test_kernel_window_misses(self):
        v = world(lambda w: 0.0 if w <= 0.375 else 0.1, w_fn=lambda w: max(0.0, PRED[w]["A"] - (0.1 if w > 0.375 else 0)))
        self.assertEqual(v["Q-2"], "the kernel window misses part of its own write-backs")

    def test_validity(self):
        self.assertFalse(world(lambda w: 0.0, ctl=5000.0)["Q-0"])
        self.assertFalse(world(lambda w: 0.0, fails=1)["Q-0"])
        self.assertFalse(world(lambda w: 0.2)["Q-0 parts"]["c below-cap sums"])  # S_k(0.375) = 0.82

    def test_v4_off(self):
        self.assertEqual(world(lambda w: 0.0, a_fn=lambda w: 0.0)["Q-3"], "v4 does not")


if __name__ == "__main__":
    unittest.main()
