import numpy as np
import pytest

from hmfit_core.acid_base import make_simple_acid_base_system
from hmfit_core.potentiometry import (PotentiometryExperiment, simulate_pH_titration,
                                      _simulate_pH_titration_reference)


@pytest.mark.parametrize('conc', [1e-12, 1e-6, 1e-2, 1.0])
@pytest.mark.parametrize('pkas', [[7.], [13., 7., 1.]])
def test_vectorized_matches_brent_extremes_and_equivalence(conc, pkas):
    system = make_simple_acid_base_system(analytical_concentration=conc, pka=pkas,
                                          base_charge=-len(pkas))
    strong = np.array([-1., -.1, -conc, -conc / 2, -1e-10, 0., 1e-10,
                       conc / 2, conc, .1, 1.])
    experiment = PotentiometryExperiment(initial_volume=10., titrant_volumes=np.linspace(0, 10, len(strong)),
                                          analyte_concentration=conc, titrant_concentration=conc)
    opts = {'strong_ion_charge': strong}
    ref = _simulate_pH_titration_reference(experiment, system, opts)
    actual = simulate_pH_titration(experiment, system, opts)
    np.testing.assert_allclose(actual, ref, atol=1e-10, rtol=0.)
    if conc == 1e-2:
        assert np.min(actual) < 2 and np.max(actual) > 12


def test_refit_parameters_agree_within_one_thousandth_se(monkeypatch):
    import hmfit_core.potentiometry as pot
    from dataclasses import replace
    system = make_simple_acid_base_system(analytical_concentration=.01, pka=[7., 4.], base_charge=-2)
    exp = PotentiometryExperiment(initial_volume=10., titrant_volumes=np.linspace(0, 35., 60),
                                  analyte_concentration=.01, titrant_concentration=.01)
    observed = _simulate_pH_titration_reference(exp, system) + np.random.default_rng(17).normal(0, .002, 60)
    exp = replace(exp, measured_pH=observed)
    fit = pot.fit_potentiometry(exp, system, initial_pka=[6.5, 4.5])
    monkeypatch.setattr(pot, 'simulate_pH_titration', _simulate_pH_titration_reference)
    ref = pot.fit_potentiometry(exp, system, initial_pka=[6.5, 4.5])
    assert fit.success and ref.success
    se = np.sqrt(np.diag(ref.covariance_matrix))
    assert np.all(np.abs(fit.theta_hat - ref.theta_hat) <= 1e-3 * se)


def test_nonmonotone_species_charges_use_reference(monkeypatch):
    import hmfit_core.potentiometry as pot
    system = make_simple_acid_base_system(analytical_concentration=.01, pka=[7.], base_charge=-1)
    system.components[0].species[1].charge = -2
    exp = PotentiometryExperiment(initial_volume=10., titrant_volumes=np.arange(3.),
                                  analyte_concentration=.01, titrant_concentration=.01)
    expected = _simulate_pH_titration_reference(exp, system)
    calls = []
    def reference(*args, **kwargs):
        calls.append(True)
        return _simulate_pH_titration_reference(*args, **kwargs)
    monkeypatch.setattr(pot, '_simulate_pH_titration_reference', reference)
    np.testing.assert_allclose(pot.simulate_pH_titration(exp, system), expected, atol=0, rtol=0)
    assert calls


def test_unbracketed_points_use_reference_fallback():
    system = make_simple_acid_base_system(analytical_concentration=.01, pka=[7.], base_charge=-1)
    exp = PotentiometryExperiment(initial_volume=10., titrant_volumes=np.arange(5.),
                                  analyte_concentration=.01, titrant_concentration=.01)
    opts = {'pH_bounds': (4., 10.), 'strong_ion_charge': [-1., -.001, 0., .001, 1.]}
    np.testing.assert_allclose(simulate_pH_titration(exp, system, opts),
                               _simulate_pH_titration_reference(exp, system, opts), atol=1e-10, rtol=0.)
