"""Unit tests for b19_score.py on synthetic data."""

import unittest

import b19_score as S

MIB = 1 << 20
BASE = {"rms16": 13.0, "gemv16": 8.0, "gemv": 40.0, "rms64": 50.0, "e_q": 15.0}


def build(device="NVIDIA H100 NVL", l2=60 * MIB, world=None, fails=0, kernels=None):
    world = world or {}
    metas = []
    for _ in range(3):
        ev = []
        for k in ("rms16", "gemv16", "gemv"):
            for c in ("plain", "pad"):
                for m in ("zero256", "zeroL2", "pr256", "none"):
                    t = world.get((k, c, m), BASE[k])
                    ev.append({"kind": "loop", "kernel": k, "cond": c, "mode": m, "times_us": [t, t, t]})
        for k in ("rms64", "rms16", "gemv", "e_q"):
            for cfg in ("cold", "steady"):
                t = world.get((k, cfg), BASE[k])
                ev += [{"kind": "bench", "kernel": k, "config": cfg, "mean_us": t}] * 5
            ev.append({"kind": "do_bench", "kernel": k, "median_us": BASE[k] * 1.2})
        metas.append({"device": device, "l2_bytes": l2, "kernels": kernels or ["e_q", "gemv", "gemv16", "rms16", "rms64"],
                      "events": ev, "correctness_failures": fails, "pads_us": {"rms16": 300.0, "gemv": 300.0},
                      "host_dispatch_us": {"rms16": 60.0}, "notes": {}})
    return metas


def predicted_world(l2=60 * MIB, r=0.431):
    rb = r * l2
    P = rb / (128 * MIB - rb)
    return {("rms16", "plain", "zeroL2"): 13.0 * 3.0, ("gemv16", "pad", "none"): 8.0 * 0.5,
            ("rms64", "steady"): 50.0 * (1 + P), ("e_q", "steady"): 15.0 * 1.7}


class TestScore(unittest.TestCase):
    def test_predicted(self):
        v = S.score(build(world=predicted_world()), log=lambda *_: None)
        self.assertTrue(v["R-0"])
        self.assertEqual(v["R-1"], "the B17 anomaly replicates")
        self.assertEqual(v["R-2"], "queue drain")
        self.assertEqual(v["R-3"], "no flush is warm once the queue is busy")
        self.assertEqual(v["R-4a"], "steady charges the deferred output write-back")
        self.assertEqual(v["R-4b"], "the residue predicts the cold/steady gap")

    def test_not_queue_drain_and_no_anomaly(self):
        w = predicted_world()
        w[("rms16", "pad", "zeroL2")] = 13.0 * 3.0
        self.assertEqual(S.score(build(world=w), log=lambda *_: None)["R-2"], "not queue drain")
        w2 = predicted_world()
        del w2[("rms16", "plain", "zeroL2")]
        v = S.score(build(world=w2), log=lambda *_: None)
        self.assertEqual(v["R-1"], "no anomaly")
        self.assertEqual(v["R-2"], "n/a")

    def test_gate_and_residue_lookup(self):
        self.assertFalse(S.score(build(world=predicted_world(), kernels=["e_q", "gemv", "gemv16"]), log=lambda *_: None)["R-0"])
        self.assertEqual(S.residue_for("NVIDIA GeForce RTX 4060 Ti"), 0.301)
        self.assertEqual(S.residue_for("NVIDIA GeForce RTX 4060"), 0.311)
        self.assertEqual(S.residue_for("NVIDIA A100-PCIE-40GB"), 0.448)
        self.assertIsNone(S.residue_for("NVIDIA GeForce RTX 3060 Laptop GPU"))

    def test_all(self):
        vs = [S.score(build(world=predicted_world()), log=lambda *_: None) for _ in range(4)]
        out = S.score_all(vs, log=lambda *_: None)
        self.assertTrue(out["R-2 all"])
        self.assertTrue(out["R-4b"].startswith("the residue predicts"))


if __name__ == "__main__":
    unittest.main()
