"""Unit tests for the shared reporting-convention helpers in hmfit_core.utils.errors."""
from __future__ import annotations

import numpy as np

from hmfit_core.utils.errors import correlation_from_covariance, enforce_uncertainty_convention


def test_correlation_from_covariance_marks_fixed_and_nonidentifiable_as_nan():
    cov = np.array([[4.0, 1.0, 0.0, np.nan], [1.0, 1.0, 0.0, np.nan], [0.0, 0.0, 0.0, 0.0],
                    [np.nan, np.nan, 0.0, np.nan]])
    corr = correlation_from_covariance(cov)
    assert corr[0, 1] == corr[1, 0] == 0.5
    assert corr[0, 0] == corr[1, 1] == 1.0
    assert np.all(np.isnan(corr[2:, :])) and np.all(np.isnan(corr[:, 2:]))
    flagged = correlation_from_covariance(cov[:2, :2], undefined=[False, True])
    assert flagged[0, 0] == 1.0 and np.all(np.isnan(flagged[1, :])) and np.isnan(flagged[0, 1])


def test_enforce_convention_fixed_nonidentifiable_and_dof():
    est = np.array([1.0, 2.0, 3.0])
    se = np.array([0.1, 1e-17, 0.3])
    lo, hi = est - 0.2, est + 0.2
    corr = np.full((3, 3), 0.1) + 0.9 * np.eye(3)
    s, l, h, c, (x16, x84), notes = enforce_uncertainty_convention(
        est, se, lo, hi, corr, fixed_mask=[False, True, False], nonidentifiable=[False, False, True], dof=5,
        extra_ci=(lo.copy(), None))
    assert s[0] == 0.1 and s[1] == 0.0 and np.isnan(s[2])
    assert l[1] == h[1] == 2.0 and np.isnan(l[2]) and np.isnan(h[2]) and x16[1] == 2.0 and x84 is None
    assert c[0, 0] == 1.0 and np.all(np.isnan(c[1:, :])) and np.all(np.isnan(c[:, 1:]))
    assert notes == []
    s, l, h, c, _, notes = enforce_uncertainty_convention(est, se, lo, hi, corr, fixed_mask=[False, True, False], dof=0)
    assert np.isnan(s[0]) and s[1] == 0.0 and np.isnan(s[2]) and np.all(np.isnan(c)) and notes
    # inputs are not modified in place
    assert se[1] == 1e-17 and corr[0, 1] == 0.1
