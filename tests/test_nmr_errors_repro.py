import numpy as np
import pytest
from hmfit_core.utils.errors import compute_errors_nmr_varpro

class MockRes:
    def __init__(self, m, n_abs, nspec):
        self.m = m
        self.n_abs = n_abs
        self.nspec = nspec
    def concentraciones(self, k):
        C = np.random.rand(self.m, self.n_abs)
        Co = np.random.rand(self.m, self.nspec)
        return C, Co

def test_nmr_errors_dof_logic():
    m = 40
    nP = 3
    n_abs = 2
    nspec = 4
    p = 2 # number of parameters k
    k = np.array([2.0, 3.0])
    
    res = MockRes(m, n_abs, nspec)
    dq = np.random.rand(m, nP)
    H = np.ones(m)
    modelo = np.random.rand(nspec, 2) # nspec x n_comp
    nas = [0, 1] # indices of non-absorbent species
    
    # Test without mask
    err_res = compute_errors_nmr_varpro(k, res, dq, H, modelo, nas, mask=None)
    assert np.isfinite(err_res["covfit"]), "s2 should be finite"
    assert not np.all(err_res["SE_log10K"] == 0), "SE should not be all zero"
    
    # Test with mask
    mask = np.ones((m, nP), dtype=bool)
    # fixed_mask where the second parameter is fixed
    fixed_mask = np.array([False, True])
    err_res_masked = compute_errors_nmr_varpro(k, res, dq, H, modelo, nas, mask=mask, fixed_mask=fixed_mask)
    assert np.isfinite(err_res_masked["covfit"]), "s2 with mask should be finite"
    assert err_res_masked["SE_log10K"][1] == 0, "Fixed parameter SE should be 0"
    assert err_res_masked["SE_log10K"][0] > 0, "Free parameter SE should be > 0"
    assert err_res_masked["dof"] == (m * nP - 1), f"dof should be {m*nP - 1}, got {err_res_masked['dof']}"

def test_nmr_errors_fallback_logic():
    """Mocks the logic in nmr_processor.py when nas is empty."""
    m = 40
    nP = 3
    p = 5 # number of free parameters
    
    # Simulate residuals_vec
    residuals_vec = np.random.normal(0, 0.01, size=(m * nP))
    
    # This is what nmr_processor.py does on lines 415-417
    N_res = residuals_vec.size
    N_par = p
    dof = N_res - N_par
    
    SS_res = float(np.sum(residuals_vec**2))
    
    # Line 459 in nmr_processor.py
    s2 = (SS_res / dof) if dof > 0 else np.nan
    
    assert np.isfinite(s2), f"s2 should be finite, got {s2} (dof={dof})"
    assert dof == 115, f"dof should be 115, got {dof}"

if __name__ == "__main__":
    test_nmr_errors_dof_logic()
    test_nmr_errors_fallback_logic()
