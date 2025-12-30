import numpy as np
from hmfit_core.utils.errors import build_identifiability_report, compute_condition_from_psd_eigs

def test_compute_condition():
    # Well conditioned
    eigs = np.array([100, 10, 1])
    assert compute_condition_from_psd_eigs(eigs) == 100
    
    # Ill conditioned
    eigs = np.array([1e8, 1])
    assert compute_condition_from_psd_eigs(eigs) == 1e8
    
    # Zero eigenvalue
    eigs = np.array([100, 0])
    assert compute_condition_from_psd_eigs(eigs) == np.inf

def test_build_identifiability_report_excellent():
    JJT = np.eye(3)
    Cov_free = np.eye(3) * 0.01
    report = build_identifiability_report(JJT, Cov_free, param_names=["K1", "K2", "K3"])
    
    assert report["status"] == "excellent"
    assert report["cond_jjt"] == 1.0
    assert report["max_abs_corr"] < 1e-12
    assert report["rank_eff"] == 3
    
    indicator = report["stability_indicator"]
    assert indicator["label"] == "Stable"
    assert indicator["icon"] == "✅"

def test_combined_rule_high_corr_low_cond():
    # cond=1.0 but correlation=0.97
    JJT = np.eye(2)
    v = 0.97
    # Covariance with high correlation
    Cov = np.array([[1.0, v], [v, 1.0]])
    report = build_identifiability_report(JJT, Cov, param_names=["K1", "K2"])
    
    # Even if cond is 1.0, maxcorr 0.97 should trigger WARNING
    assert report["status"] == "warn"
    assert "high correlation" in report["stability_indicator"]["reasons"]
    assert "Warning" in report["stability_indicator"]["label"]

def test_combined_rule_critical_corr():
    JJT = np.eye(2)
    v = 0.995
    Cov = np.array([[1.0, v], [v, 1.0]])
    report = build_identifiability_report(JJT, Cov)
    
    # maxcorr >= 0.99 should trigger CRITICAL
    assert report["status"] == "critical"
    assert "Ill-conditioned" in report["stability_indicator"]["label"]

def test_combined_rule_singular():
    # One eigenvalue is zero
    JJT = np.array([[1.0, 0.0], [0.0, 0.0]])
    Cov = np.eye(2) # doesn't matter much for this test
    report = build_identifiability_report(JJT, Cov)
    
    assert report["status"] == "critical"
    assert report["rank_eff"] == 1
    assert report["p_free"] == 2
    assert "Singular" in report["stability_indicator"]["label"]
    assert "singular" in report["stability_indicator"]["reasons"]

def test_build_identifiability_report_warn_cond():
    # cond=1e5 should trigger WARNING
    JJT = np.diag([1e5, 1.0])
    Cov = np.eye(2)
    report = build_identifiability_report(JJT, Cov)
    
    assert report["status"] == "warn"
    assert "poor conditioning" in report["stability_indicator"]["reasons"]
