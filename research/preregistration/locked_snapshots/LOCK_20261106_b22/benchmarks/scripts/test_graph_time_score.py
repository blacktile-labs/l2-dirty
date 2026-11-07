"""Unit tests for graph_time_score.py (B22) on synthetic data."""
import unittest

import graph_time_score as S

KT = {"qkv": 60.0, "o": 20.0, "gate_up": 110.0, "down": 55.0}


def build(graph_err=0.005, cold_err=0.10, int4_graph_err=None, fails=0, noisy=False):
    metas = []
    for _ in range(3):
        ev = []
        for v, sp in (("fp16", 1.0), ("int4", 3.0)):
            t = 4 * sum(KT.values()) / sp
            spread = 0.05 if noisy else 0.001
            ev.append({"kind": "step", "variant": v, "times_us": [t * (1 - spread), t, t * (1 + spread)]})
            ge = graph_err if (v == "fp16" or int4_graph_err is None) else int4_graph_err
            for s, k in KT.items():
                c = k / sp
                ev.append({"kind": "kernel", "variant": v, "shape": s, "graph_us": c * (1 + ge), "cold_us": c * (1 + cold_err),
                           "write_us": c * 1.3})
        metas.append({"device": "NVIDIA H100 NVL", "l2_bytes": 60 << 20, "layers": 4, "events": ev,
                      "correctness_failures": fails, "ncopies": {}, "notes": {}})
    return metas


class T(unittest.TestCase):
    def q(self, **kw):
        return S.score(build(**kw), log=lambda *_: None)

    def test_within_1(self):
        v = self.q()
        self.assertEqual(v["V-1"], "within 1 %")
        self.assertEqual(v["V-2"], "speedup within 1 %")
        self.assertEqual(v["V-3"], "graph timing beats cold")

    def test_within_2_and_not(self):
        self.assertEqual(self.q(graph_err=0.015)["V-1"], "within 2 %")
        self.assertEqual(self.q(graph_err=0.05)["V-1"], "not")
        self.assertEqual(self.q(graph_err=0.0, int4_graph_err=0.015)["V-2"], "speedup within 2 %")
        self.assertEqual(self.q(graph_err=0.2)["V-3"], "not")

    def test_gate(self):
        self.assertFalse(self.q(fails=1)["V-0"])
        self.assertFalse(self.q(noisy=True)["V-0"])


if __name__ == "__main__":
    unittest.main()
