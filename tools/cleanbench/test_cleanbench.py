"""CPU-only tests of cleanbench's statistics helpers (no GPU needed)."""
import unittest

import cleanbench as cb


class T(unittest.TestCase):
    def test_tick_fraction(self):
        self.assertEqual(cb.tick_fraction([1.024, 2.048, 3.072]), 1.0)
        self.assertEqual(cb.tick_fraction([1.5, 2.5]), 0.0)

    def test_quantiles(self):
        self.assertEqual(cb.quantiles([1.0, 2.0, 3.0, 4.0, 5.0])[:3], (2.0, 3.0, 4.0))

    def test_warnings(self):
        self.assertTrue(any("timer ticks" in w for w in cb.warnings_for([10.24] * 20, "cold")))
        self.assertTrue(any("IQR" in w for w in cb.warnings_for([80.0, 100.0, 120.0, 140.0], "steady")))

    def test_records(self):
        r = [("warm", 1, 5), ("_cb_marker", 2, 1), ("k", 3, 1000), ("j", 4, 500), ("k", 5, 2000), ("j", 6, 500)]
        rec = cb.after_marker(r)
        self.assertEqual(len(rec), 4)
        self.assertTrue(cb.records_ok(rec, ["k", "j"], 2))
        self.assertFalse(cb.records_ok(rec, ["k", "j"], 3))
        self.assertFalse(cb.records_ok(rec, [], 0))
        self.assertEqual(cb.call_times(rec, 2), [1.5, 2.5])
        self.assertEqual(cb.call_times(rec, 3), [])
        self.assertIsNone(cb.after_marker([("k", 1, 1)]))

    def test_quality_gates(self):
        L2 = 40 << 20
        ok = cb.quality_gates(True, 0.001, {"SW Thermal Slowdown": 5}, {"SW Thermal Slowdown": 5}, 4 * L2, L2)
        self.assertTrue(ok["validated"])
        self.assertFalse(cb.quality_gates(True, 0.02, {"a": 0}, {"a": 0}, 4 * L2, L2)["validated"])        # tracing
        slow = cb.quality_gates(True, 0.0, {"SW Power Capping": 1}, {"SW Power Capping": 9}, 4 * L2, L2)
        self.assertFalse(slow["clock_ok"])
        self.assertEqual(slow["clock_slowdown_us"], {"SW Power Capping": 8})
        self.assertFalse(cb.quality_gates(True, 0.0, {}, {}, 4 * L2, L2)["validated"])                  # no counters
        self.assertIsNone(cb.quality_gates(True, 0.0, {"a": 0}, {"a": 0}, None, L2)["footprint_ok"])
        self.assertFalse(cb.quality_gates(True, 0.0, {"a": 0}, {"a": 0}, None, L2)["validated"])        # unknown footprint
        self.assertFalse(cb.quality_gates(True, 0.0, {"a": 0}, {"a": 0}, 2 * L2, L2)["validated"])      # too small
        self.assertFalse(cb.quality_gates(False, 0.0, {"a": 0}, {"a": 0}, 4 * L2, L2)["validated"])     # records


if __name__ == "__main__":
    unittest.main()
