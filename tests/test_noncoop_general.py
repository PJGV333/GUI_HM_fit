import os
import sys
import unittest

import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class TestNonCooperativeGeneral(unittest.TestCase):
    def test_1_to_3_single_k1_out_of_order_columns(self):
        from NR_conc_algoritm import NewtonRaphson
        from noncoop_utils import noncoop_derived_from_logK1

        # 2 components, 3 complexes (N=3)
        # Columns: [H, G, HG3, HG, HG2] (out of order)
        modelo = np.array(
            [
                [1, 0, 1, 1, 1],
                [0, 1, 3, 1, 2],
            ],
            dtype=float,
        )
        ctot = np.array([[1.0, 1.0]], dtype=float)
        res = NewtonRaphson(ctot, modelo, nas=[], model_sett="Non-cooperative")

        logK1 = 3.0
        k_num = res._prepare_K_numeric([logK1])
        log_beta = np.log10(k_num)

        # Expected stepwise:
        # logK2 = logK1 - log10(3)
        # logK3 = logK1 - log10(9)
        logK2 = logK1 - np.log10(3.0)
        logK3 = logK1 - np.log10(9.0)
        beta1 = logK1
        beta2 = logK1 + logK2
        beta3 = logK1 + logK2 + logK3

        # Components
        self.assertAlmostEqual(log_beta[0], 0.0, places=12)
        self.assertAlmostEqual(log_beta[1], 0.0, places=12)
        # Complexes mapped by j (HG3 at col 2, HG at col 3, HG2 at col 4)
        self.assertAlmostEqual(log_beta[3], beta1, places=12)  # HG (j=1)
        self.assertAlmostEqual(log_beta[4], beta2, places=12)  # HG2 (j=2)
        self.assertAlmostEqual(log_beta[2], beta3, places=12)  # HG3 (j=3)

        derived = noncoop_derived_from_logK1(modelo, logK1)
        self.assertEqual(int(derived["N"]), 3)
        self.assertAlmostEqual(float(derived["logK_by_j"][1]), logK2, places=12)
        self.assertAlmostEqual(float(derived["logK_by_j"][2]), logK3, places=12)

    def test_invalid_noncoop_requires_series(self):
        from LM_conc_algoritm import LevenbergMarquardt

        # 2 components, but complexes are not a pure 1:N or N:1 series
        modelo = np.array(
            [
                [1, 0, 2, 1],
                [0, 1, 1, 2],
            ],
            dtype=float,
        )
        ctot = np.array([[1.0, 1.0]], dtype=float)
        res = LevenbergMarquardt(ctot, modelo, nas=[], model_sett="Non-cooperative")

        with self.assertRaises(ValueError):
            res._prepare_K_numeric([3.0])


if __name__ == "__main__":
    unittest.main()
