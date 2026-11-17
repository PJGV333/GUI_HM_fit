import numpy as np
from hmfit_core.solvers import NewtonRaphson


def test_independent_12_equilibrium_matches_mass_balance_and_production():
    from validation.stage_e3_identifiability import equilibrium_12
    totals = np.column_stack((np.full(10, 1e-3), np.linspace(0, 2e-3, 10)))
    constants = np.array([3., 2.5])
    C = equilibrium_12(totals, constants)
    np.testing.assert_allclose(C[:,0]+C[:,2]+C[:,3], totals[:,0], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(C[:,1]+C[:,2]+2*C[:,3], totals[:,1], rtol=1e-10, atol=1e-14)
    solver = NewtonRaphson(totals, np.array([[1,0,1,1],[0,1,1,2]]), nas=[], model_sett='Free')
    reference, _ = solver.concentraciones(constants)
    np.testing.assert_allclose(C, reference, rtol=1e-7, atol=1e-12)


def test_independent_bounded_reference_matches_unconstrained_lstsq():
    from validation.stage_e3_identifiability import bounded_reference
    C = np.array([[1., 0.], [0., 1.], [1., 1.]])
    Y = np.array([[1., 3., 5.], [2., 4., 6.], [3., 7., 11.]])
    A, residual, kkt = bounded_reference(C, Y, lower=np.array([-100., -100.]), alpha=0.)
    np.testing.assert_allclose(A, np.linalg.lstsq(C, Y, rcond=None)[0], atol=1e-12)
    assert np.linalg.norm(residual) < 1e-12
    assert kkt < 1e-12


def test_equilibrium_sensitivity_matches_production_at_two_constants():
    from validation.stage_e3_identifiability import equilibrium_12
    totals = np.column_stack((np.full(10, 1e-3), np.linspace(0, 2e-3, 10)))
    solver = NewtonRaphson(totals, np.array([[1,0,1,1],[0,1,1,2]]), nas=[], model_sett='Free')
    truth = np.array([3., 2.5])
    for direction in np.eye(2)*1e-3:
        prod = solver.concentraciones(truth+direction)[0] - solver.concentraciones(truth-direction)[0]
        ref = equilibrium_12(totals, truth+direction) - equilibrium_12(totals, truth-direction)
        np.testing.assert_allclose(prod, ref, rtol=1e-5, atol=1e-13)


def test_profile_reoptimizes_the_nuisance_constant():
    from validation.stage_e3_identifiability import equilibrium_12, profile_coordinate
    totals = np.column_stack((np.full(10, 1e-3), np.linspace(0, 2e-3, 10)))
    wl = np.linspace(240., 600., 31)
    eps = np.vstack([1200*np.exp(-((wl-center)**2)/(2*25.**2)) for center in np.linspace(265.,380.,4)])
    Y = equilibrium_12(totals, [3.,2.5]) @ eps
    result = profile_coordinate(totals, Y, 0., 1, 2.5)
    assert abs(result['nuisance']-3.) < .01
    assert result['objective'] < 1e-10
    assert result['kkt_abs'] < 1e-8
