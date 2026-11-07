"""Unit tests for cupti_records.py (B23 record segmentation) on synthetic records."""

import unittest

import cupti_records as C


def rec(name, dur_ns, t=[0]):  # noqa: B006 -- running timestamp
    t[0] += 1000
    return (name, t[0], dur_ns)


class T(unittest.TestCase):
    def test_after_marker(self):
        r = [rec("warm", 5), rec("_b23_marker", 1), rec("a", 2), rec("_b23_marker", 1), rec("b", 3)]
        self.assertEqual([x[0] for x in C.after_marker(r)], ["b"])        # last marker wins
        self.assertIsNone(C.after_marker([rec("a", 1)]))
        self.assertEqual(C.after_marker([rec("x", 1), rec("_b23_marker", 1)]), [])

    def test_records_ok(self):
        exp = ["k1", "k2"]
        good = [rec(n, 1) for _ in range(3) for n in exp]
        self.assertTrue(C.records_ok(good, exp, 3))
        self.assertFalse(C.records_ok(good, exp, 2))                       # count
        self.assertFalse(C.records_ok(good[:-1] + [rec("k9", 1)], exp, 3))  # name
        self.assertFalse(C.records_ok([], [], 0))                           # empty expectation never passes
        self.assertFalse(C.records_ok(None, exp, 3))
        long = "x" * 300
        self.assertTrue(C.records_ok([rec(long, 1)], [long[:C.NAME_LEN]], 1))

    def test_op_times(self):
        # two repetitions of ops with 1 and 2 kernels: durations summed per op, in us
        r = [rec("a", 1000), rec("b", 2000), rec("c", 500), rec("a", 3000), rec("b", 100), rec("c", 100)]
        self.assertEqual(C.op_times(r, [1, 2]), [[1.0, 3.0], [2.5, 0.2]])
        self.assertEqual(C.op_times(r, [0, 2]), [[], []])                  # no infinite loop on k = 0
        self.assertEqual(C.op_times(r[:-1], [1, 2]), [[], []])             # partial repetition rejected
        self.assertEqual(C.op_times([], [1]), [[]])


if __name__ == "__main__":
    unittest.main()
