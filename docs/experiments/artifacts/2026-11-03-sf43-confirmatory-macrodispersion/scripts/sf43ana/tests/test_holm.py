import unittest

import sf43ana  # noqa: F401
from sf43ana.holm import holm


def H(i, p, testable=True, exact=None):
    return {"id": i, "p": p, "testable": testable, "exact": exact}


class HolmTests(unittest.TestCase):
    def test_textbook(self):
        res, info = holm([H("a", 0.001), H("b", 0.01), H("c", 0.04), H("d", 0.03)], 0.05)
        self.assertEqual([r["rejected"] for r in res], [True, True, False, False])  # .001<=.0125, .01<=.0167, .03>.025: stop
        self.assertEqual([r["rank"] for r in res], [1, 2, 4, 3])
        self.assertAlmostEqual(res[0]["level"], 0.05 / 4)
        self.assertAlmostEqual(res[1]["p_holm_adjusted"], 0.03)
        self.assertEqual(info["n_rejected"], 2)

    def test_untestable_enters_with_p_one_and_keeps_m(self):
        # with the fixed family of 4, the first level is 0.05/4 even though one hypothesis cannot be tested
        res, info = holm([H("a", 0.0124), H("b", 0.5, testable=False), H("c", 0.9), H("d", 0.01)], 0.05)
        self.assertEqual(info["m"], 4)
        self.assertEqual([r["rejected"] for r in res], [True, False, False, True])
        self.assertTrue(res[1]["p"] == 1.0 and res[1]["decided_by"] == "untestable")  # the supplied 0.5 is ignored
        res2, _ = holm([H("a", 0.0126), H("b", 1.0, testable=False), H("c", 1.0, testable=False), H("d", 1.0, testable=False)], 0.05)
        self.assertFalse(res2[0]["rejected"])  # 0.0126 > 0.05/4: dropping the untestable ones would have rejected it

    def test_exact_decision_at_the_level_and_stop_rule(self):
        calls = []

        def ex(v):
            def f(level):
                calls.append((v, level))
                return v

            return f

        # p slightly above the level on the lattice (0.0170 > 0.05/3), but the exact limit at the level rejects
        res, _ = holm([H("a", 0.0170, exact=ex(True)), H("b", 0.5, exact=ex(False)), H("c", 0.9, exact=ex(True))], 0.05)
        self.assertTrue(res[0]["rejected"] and not res[0]["lattice_rejected"] and res[0]["decided_by"] == "exact")
        self.assertEqual([r["rejected"] for r in res], [True, False, False])  # c is not reached after b fails, though exact says True
        self.assertEqual([c[1] for c in calls], [0.05 / 3, 0.05 / 2])

    def test_no_rejection_after_first_failure(self):
        res, _ = holm([H("a", 0.2, exact=lambda lv: False), H("b", 0.3, exact=lambda lv: True)], 0.05)
        self.assertEqual([r["rejected"] for r in res], [False, False])

    def test_all_untestable(self):
        res, info = holm([H("a", 1.0, testable=False), H("b", 1.0, testable=False)], 0.05)
        self.assertEqual(info["n_rejected"], 0)


if __name__ == "__main__":
    unittest.main()
