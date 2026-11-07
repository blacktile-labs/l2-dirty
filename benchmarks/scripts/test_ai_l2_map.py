import unittest

import ai_l2_map as m


class AiL2Map(unittest.TestCase):
    def test_zone_boundaries(self):
        l2 = 4 * m.MiB
        self.assertEqual(m.zone(2 * m.MiB, l2), "L2")          # == 0.5 L2
        self.assertEqual(m.zone(2 * m.MiB + 1, l2), "edge")
        self.assertEqual(m.zone(4 * m.MiB, l2), "edge")        # == L2
        self.assertEqual(m.zone(4 * m.MiB + 1, l2), "DRAM")

    def test_tensor_sizes_gqa_swiglu(self):
        t = m.tensors(4096, 32, 8, 14336, 2048)
        self.assertEqual(t["T x d (norm, attn-out, residual)"], 2048 * 4096 * 2)
        self.assertEqual(t["qkv"], 2048 * (4096 + 2 * 1024) * 2)
        self.assertEqual(t["gate+up (T x 2 d_ff)"], 2048 * 2 * 14336 * 2)

    def test_batch1_decode_fits_everywhere(self):
        rows = [r for r in m.build() if r["phase"] == "decode" and r["T"] == 1]
        self.assertTrue(rows)
        self.assertTrue(all(z == "L2" for r in rows for z in r["zones"].values()))


if __name__ == "__main__":
    unittest.main()
