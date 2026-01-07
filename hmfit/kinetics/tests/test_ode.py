"""Tests for kinetic ODE integration."""

from __future__ import annotations

import numpy as np

from hmfit.kinetics.data.dataset import KineticsDataset
from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
from hmfit.kinetics.model.kinetics_model import KineticsModel


def test_first_order_solution_matches_analytic() -> None:
    text = """
    species: A, B
    A -> B ; k1
    """
    model = KineticsModel(parse_mechanism(text))

    t = np.linspace(0.0, 5.0, 101)
    y0 = {"A": 1.0, "B": 0.0}
    params = {"k1": 0.7}
    dataset = KineticsDataset(t=t, y0=y0, fixed_conc={}, temperature=298.15)

    conc = model.solve_concentrations(t, y0, params, dataset)
    a_num = conc[:, model.species_index["A"]]
    b_num = conc[:, model.species_index["B"]]

    a_anal = y0["A"] * np.exp(-params["k1"] * t)
    b_anal = y0["A"] - a_anal

    assert np.max(np.abs(a_num - a_anal)) < 1e-4
    assert np.max(np.abs(b_num - b_anal)) < 1e-4
    assert np.max(np.abs(a_num + b_num - y0["A"])) < 1e-4


def test_consecutive_reaction_conserves_mass() -> None:
    text = """
    species: A, B, C
    A -> B ; k1
    B -> C ; k2
    """
    model = KineticsModel(parse_mechanism(text))

    t = np.linspace(0.0, 10.0, 201)
    y0 = {"A": 1.0, "B": 0.0, "C": 0.0}
    params = {"k1": 1.0, "k2": 0.3}
    dataset = KineticsDataset(t=t, y0=y0, fixed_conc={}, temperature=298.15)

    conc = model.solve_concentrations(t, y0, params, dataset)
    a_num = conc[:, model.species_index["A"]]
    b_num = conc[:, model.species_index["B"]]
    c_num = conc[:, model.species_index["C"]]

    assert a_num[-1] < a_num[0]
    assert c_num[-1] > c_num[0]
    assert np.all(a_num >= -1e-10)
    assert np.all(b_num >= -1e-10)
    assert np.all(c_num >= -1e-10)
    assert np.max(np.abs(a_num + b_num + c_num - y0["A"])) < 1e-4


def test_reversible_reaction_equilibrium() -> None:
    text = """
    species: A, B
    A <-> B ; kf, kr
    """
    model = KineticsModel(parse_mechanism(text))

    t = np.linspace(0.0, 30.0, 301)
    y0 = {"A": 1.0, "B": 0.0}
    params = {"kf": 2.0, "kr": 1.0}
    dataset = KineticsDataset(t=t, y0=y0, fixed_conc={}, temperature=298.15)

    conc = model.solve_concentrations(t, y0, params, dataset)
    a_end = conc[-1, model.species_index["A"]]
    b_end = conc[-1, model.species_index["B"]]

    assert abs(a_end - 1.0 / 3.0) < 1e-3
    assert abs(b_end - 2.0 / 3.0) < 1e-3


def test_temperature_callable_runs() -> None:
    text = """
    species: A, B
    A -> B ; k1
    """
    model = KineticsModel(parse_mechanism(text))

    t = np.linspace(0.0, 2.0, 21)
    y0 = {"A": 1.0, "B": 0.0}
    params = {"k1": 0.5}
    dataset = KineticsDataset(
        t=t,
        y0=y0,
        fixed_conc={},
        temperature=lambda time: 298.15 + 0.0 * time,
    )

    conc = model.solve_concentrations(t, y0, params, dataset)

    assert conc.shape == (t.shape[0], 2)
