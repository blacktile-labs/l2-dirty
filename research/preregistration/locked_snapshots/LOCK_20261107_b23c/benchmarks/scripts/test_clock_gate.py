"""Unit tests for clock_gate.py (B23b) on synthetic nvidia-smi output."""

import unittest

import clock_gate as C

PERF = """    Clocks Event Reasons Counters
        SW Power Capping                               : {pc} us
        Sync Boost                                     : 0 us
        SW Thermal Slowdown                            : {st} us
        HW Thermal Slowdown                            : 0 us
        HW Power Braking                               : 0 us
    Sparse Operation Mode                              : N/A
"""


def perf(pc=0, st=0):
    return PERF.format(pc=pc, st=st)


def rows(sm, util=100, temp=60):
    return [{"utilization.gpu [%]": f"{util} %", "clocks.current.sm [MHz]": f"{x} MHz", "temperature.gpu": str(temp),
             "clocks_event_reasons.active": "0x1"} for x in sm]


class T(unittest.TestCase):
    def test_counters(self):
        c = C.counters(perf(pc=5, st=7))
        self.assertEqual(c["SW Power Capping"], 5)
        self.assertEqual(c["SW Thermal Slowdown"], 7)
        self.assertEqual(c["HW Power Braking"], 0)
        self.assertEqual(C.counters("no section"), {})

    def test_steady(self):
        g = C.gate(rows([1005] * 200), [(perf(), perf(pc=100))], 1000)
        self.assertTrue(g["C-1"] and g["C-2"])
        self.assertEqual(g["counter increase us"]["SW Power Capping"], 100)    # descriptive only

    def test_clock_moves(self):
        self.assertFalse(C.gate(rows([1005] * 196 + [1747] * 4), [(perf(), perf())], 1000)["C-1"])   # 2 % off
        self.assertTrue(C.gate(rows([1005] * 198 + [1747] * 2), [(perf(), perf())], 1000)["C-1"])    # 1 % off
        self.assertFalse(C.gate(rows([1350] * 200), [(perf(), perf())], 1000)["C-1"])                # not the lock
        self.assertFalse(C.gate(rows([1005] * 10, util=50), [(perf(), perf())], 1000)["C-1"])         # no busy samples

    def test_unlocked(self):
        self.assertTrue(C.gate(rows([1350] * 200), [(perf(), perf())], None)["C-1"])                 # steady, no lock
        self.assertFalse(C.gate(rows([1350] * 190 + [1300] * 10), [(perf(), perf())], None)["C-1"])  # 5 % at -3.7 %

    def test_thermal(self):
        self.assertFalse(C.gate(rows([1005] * 200), [(perf(), perf(st=1))], 1000)["C-2"])
        self.assertFalse(C.gate(rows([1005] * 200), [(perf(st=50), perf(st=3))], 1000)["C-2"])        # reset counts
        self.assertTrue(C.gate(rows([1005] * 200), [(perf(st=50), perf(st=50))], 1000)["C-2"])
        self.assertFalse(C.gate(rows([1005] * 200), [], 1000)["C-2"])                                   # no logs


if __name__ == "__main__":
    unittest.main()
