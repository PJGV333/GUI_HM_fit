import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class _DummyAlgo:
    def __init__(self, c_t_df, modelo, nas, model_settings):
        self._n_i = int(getattr(c_t_df, "shape", [0])[0])

    def concentraciones(self, k_full):
        k_full = np.asarray(k_full, dtype=float).ravel()
        c = np.tile(k_full, (self._n_i, 1))
        co = np.zeros_like(c)
        return c, co


def _fake_project(dq_block, c_block, d_cols, mask_block, *args, **kwargs):
    dq_block = np.asarray(dq_block, dtype=float)
    c_block = np.asarray(c_block, dtype=float)
    m_sig = dq_block.shape[1]
    if c_block.shape[1] < m_sig:
        pad = np.zeros((c_block.shape[0], m_sig - c_block.shape[1]), dtype=float)
        c_block = np.concatenate([c_block, pad], axis=1)
    return c_block[:, :m_sig]


def _fake_minimize(fun, x0, method=None, bounds=None):
    x0 = np.asarray(x0, dtype=float).ravel()

    class _Res:
        pass

    res = _Res()
    res.x = x0 + 1.0
    res.success = True
    res.message = "stub"
    res.nfev = 1
    return res


class TestNmrFixedParams(unittest.TestCase):
    def _run(self, *, fixed_list=None, bounds=None, nas=None):
        from backend_fastapi.nmr_processor import process_nmr_data

        conc = pd.DataFrame(
            {
                "H": [1.0, 1.1, 1.2, 1.3, 1.4],
                "G": [0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )
        shifts = pd.DataFrame(
            {
                "sig1": [0.0, 0.0, 0.0, 0.0, 0.0],
                "sig2": [0.0, 0.0, 0.0, 0.0, 0.0],
                "sig3": [0.3, 0.3, 0.3, 0.3, 0.3],
            }
        )

        def fake_read_excel(file_path, sheet_name=None, header=0, *args, **kwargs):
            if sheet_name == "shifts":
                return shifts.copy()
            if sheet_name == "conc":
                return conc.copy()
            raise ValueError(f"Unexpected sheet_name: {sheet_name}")

        model_matrix = [
            [1, 0],
            [0, 1],
        ]

        k_initial = [0.0, 0.0, 0.3]
        if bounds is None:
            bounds = [
                {"min": -10, "max": 10},
                {"min": -10, "max": 10},
                {"min": -10, "max": 10},
            ]

        with (
            patch("backend_fastapi.nmr_processor.LevenbergMarquardt", _DummyAlgo),
            patch("backend_fastapi.nmr_processor.NewtonRaphson", _DummyAlgo),
            patch("backend_fastapi.nmr_processor.project_coeffs_block_onp_frac", _fake_project),
            patch("backend_fastapi.nmr_processor.optimize.minimize", _fake_minimize),
            patch("backend_fastapi.nmr_processor.pd.read_excel", fake_read_excel),
        ):
            return process_nmr_data(
                file_path="dummy.xlsx",
                spectra_sheet="shifts",
                conc_sheet="conc",
                column_names=["H", "G"],
                signal_names=["sig1", "sig2", "sig3"],
                receptor_label="H",
                guest_label="G",
                model_matrix=model_matrix,
                k_initial=k_initial,
                k_bounds=bounds,
                algorithm="Levenberg-Marquardt",
                optimizer="powell",
                model_settings="Free",
                non_absorbent_species=list(nas or []),
                k_fixed=fixed_list,
            )

    def test_fixed_list_keeps_constant(self):
        result = self._run(fixed_list=[False, False, True])
        self.assertTrue(result.get("success"), result)

        export = result.get("export_data") or {}
        k = export.get("k") or []
        fixed_mask = export.get("fixed_mask") or []
        se = export.get("SE_log10K") or []
        perc = export.get("percK") or []

        self.assertEqual(len(k), 3)
        self.assertEqual(k[2], 0.3)
        self.assertTrue(bool(fixed_mask[2]))
        self.assertEqual(se[2], 0.0)
        self.assertEqual(perc[2], 0.0)
        self.assertIn("± const", result.get("results_text", ""))

        constants = result.get("constants") or []
        self.assertTrue(constants[2]["fixed"])
        self.assertEqual(constants[2]["SE_log10K"], 0.0)
        self.assertEqual(constants[2]["percent_error"], 0.0)

    def test_min_eq_max_keeps_constant_without_fixed_list(self):
        bounds = [
            {"min": -10, "max": 10},
            {"min": -10, "max": 10},
            {"min": 0.3, "max": 0.3},
        ]
        result = self._run(fixed_list=None, bounds=bounds)
        self.assertTrue(result.get("success"), result)

        export = result.get("export_data") or {}
        k = export.get("k") or []
        fixed_mask = export.get("fixed_mask") or []
        se = export.get("SE_log10K") or []
        perc = export.get("percK") or []

        self.assertEqual(len(k), 3)
        self.assertEqual(k[2], 0.3)
        self.assertTrue(bool(fixed_mask[2]))
        self.assertEqual(se[2], 0.0)
        self.assertEqual(perc[2], 0.0)
        self.assertIn("± const", result.get("results_text", ""))

    def test_varpro_error_path_respects_fixed_mask(self):
        def fake_compute_errors_nmr_varpro(k, res, dq, h, modelo, nas, rcond=1e-10, use_projector=True, mask=None):
            k = np.asarray(k, dtype=float).ravel()
            p = k.size
            m_eff = int(np.sum(np.asarray(mask, dtype=bool)))
            j = np.zeros((p, m_eff), dtype=float)
            for i in range(p):
                if m_eff:
                    j[i, i % m_eff] = 1.0
            return {"J": j, "rms": 2.0, "covfit": 0.0}

        with patch("backend_fastapi.nmr_processor.compute_errors_nmr_varpro", fake_compute_errors_nmr_varpro):
            result = self._run(fixed_list=[False, False, True], nas=[0])

        self.assertTrue(result.get("success"), result)
        export = result.get("export_data") or {}
        self.assertEqual(export.get("SE_log10K")[2], 0.0)
        self.assertEqual(export.get("percK")[2], 0.0)


if __name__ == "__main__":
    unittest.main()
