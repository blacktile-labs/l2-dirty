import unittest

import flush_bias_score as s

PTX_OK = {"ld_volatile_global": 1, "ld_global": 1, "st_global": 0, "buffer_unchanged": True}


def row(fam, case, cand, zero, pr, sm, none=None, l2w=None):
    ms = {"zero": zero, "pr": pr, "sum": sm}
    if l2w is not None:
        ms["l2w"] = l2w
    if none is not None:
        ms["none"] = none
    return {"family": fam, "case": case, "cand": cand, "ms": ms}


def meta(l2_mib=64, cc="8.9", bias=0.2, dev="GPU X", ptx=PTX_OK, evict=0.5):
    """Additive write-flush penalty of `bias` x cuBLAS time on the 112 MiB shape."""
    pen = 0.200 * bias
    rows = [row("gemv", "4096x4096", "cublas_fp16", 0.050 + pen, 0.050, 0.051, none=0.050 * evict)]
    for s_ in ("14336x4096", "4096x14336"):
        rows += [row("gemv", s_, "cublas_fp16", 0.200 + pen, 0.200, 0.201, l2w=0.200 + 2 * pen),
                 row("gemv", s_, "tinygemm_k4", 0.060 + pen, 0.060, 0.061),
                 row("gemv", s_, "tinygemm_k8", 0.062 + pen, 0.062, 0.063),
                 row("gemv", s_, "bnb_nf4", 0.080 + pen, 0.080, 0.081)]
    rows += [row("gemv", "4096x4096", "tinygemm_k4", 0.016 + pen, 0.016, 0.016),
             row("gemv", "4096x4096", "bnb_nf4", 0.024 + pen, 0.024, 0.024),
             row("attn", "L65536", "sdpa_bf16", 0.300 + pen, 0.300, 0.300),
             row("rmsnorm", "T8192", "liger_rmsnorm", 0.150 + pen, 0.150, 0.150),
             row("rmsnorm", "T8192", "torch_rmsnorm", 0.400, 0.400, 0.400),
             row("swiglu", "T2048", "liger_swiglu", 0.250, 0.250, 0.250)]
    return {"device": dev, "l2_bytes": l2_mib << 20, "cc": cc, "ptx_check": ptx, "rows": rows}


class FlushBiasScore(unittest.TestCase):
    def g(self, runs=5, **kw):
        return s.score_gpu([meta(**kw)] * runs, log=lambda *_: None)

    def test_affected_distorted(self):
        v = self.g(bias=0.2)
        self.assertTrue(v["E-0"]["pass"])
        self.assertEqual(v["F-1"], "affected")
        self.assertAlmostEqual(v["B_ref"], 0.2, places=3)
        self.assertEqual(v["F-3"], "distorted")
        self.assertEqual(v["F-3 sign"], "under-estimated")
        self.assertEqual(v["F-4"]["attn"]["verdict"], "affected")       # 0.04 / 0.3 = 0.133
        self.assertEqual(v["F-4"]["swiglu"]["verdict"], "not affected")
        self.assertAlmostEqual(v["F-6"]["median R"], 0.0161, places=3)   # median over 9 gemv cases >= 38 us

    def test_f7_l2_sized_flush(self):
        v = self.g(bias=0.2)
        self.assertAlmostEqual(v["B_l2"], 0.4, places=3)
        self.assertEqual(v["F-7"], "biases more than do_bench")
        self.assertEqual(self.g(bias=0.0)["F-7"], "similar")

    def test_unaffected(self):
        v = self.g(bias=0.0)
        self.assertEqual(v["F-1"], "not affected")
        self.assertEqual(v["F-3"], "faithful")

    def test_e0_failures(self):
        self.assertFalse(self.g(runs=3)["E-0"]["pass"])
        self.assertFalse(self.g(ptx={"ld_volatile_global": 1, "st_global": 1, "buffer_unchanged": True})["E-0"]["pass"])
        self.assertFalse(self.g(evict=0.95)["E-0"]["pass"])              # PR flush does not evict
        v = self.g(l2_mib=32, evict=0.95)                                  # rule not applicable below 64 MiB
        self.assertIsNone(v["E-0"]["evict_ok"])
        self.assertTrue(v["E-0"]["pass"])

    def test_e0_amended(self):
        v = self.g(evict=0.95)                       # locked residency clause fails ...
        self.assertFalse(v["E-0"]["pass"])
        self.assertTrue(v["E-0"]["pass_amended"])   # ... but pr (0.050) is as cold as sum (0.051)
        m = meta()
        m["rows"][0]["ms"]["pr"] = 0.040             # pr clearly faster than sum: does not evict
        self.assertFalse(s.score_gpu([m] * 5, log=lambda *_: None)["E-0"]["pass_amended"])

    def test_one_missing_run_tolerated(self):
        ms = [meta()] * 4 + [dict(meta(), rows=[])]
        self.assertEqual(s.score_gpu(ms, log=lambda *_: None)["F-1"], "affected")

    def test_score_all_rho_and_replication(self):
        gpus = [self.g(l2_mib=l2, bias=b, dev=d) for l2, b, d in
                [(24, 0.05, "RTX 4060"), (32, 0.07, "RTX 4060 Ti"), (64, 0.15, "NVIDIA GeForce RTX 4080 SUPER"),
                 (72, 0.17, "RTX 4090"), (96, 0.22, "L40S"), (40, 0.30, "A100")]]
        gpus[-1]["cc"] = "8.0"
        out = s.score_all(gpus, log=lambda *_: None)
        self.assertEqual(out["F-2"], "scales with L2 within Ada")
        self.assertEqual(out["F-2 (E-0')"], "scales with L2 within Ada")
        self.assertEqual(out["F-2 rho"], 1.0)
        self.assertIn(out["F-5"], ("replicates", "differs"))
        flat = s.score_all([self.g(l2_mib=l2, bias=b) for l2, b in [(24, 0.2), (32, 0.1), (64, 0.2), (96, 0.1)]],
                           log=lambda *_: None)
        self.assertEqual(flat["F-2"], "not explained by L2 size")

    def test_spearman_ties(self):
        self.assertAlmostEqual(s.spearman([1, 2, 3], [1, 1, 2]), 0.866, places=3)


if __name__ == "__main__":
    unittest.main()
