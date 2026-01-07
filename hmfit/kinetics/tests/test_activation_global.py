"""Tests for temperature-dependent global fits."""

from __future__ import annotations

import numpy as np

from hmfit.kinetics.fit.objective import GlobalKineticsObjective
from hmfit.kinetics.fit.optimizer import fit_global
from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
from hmfit.kinetics.model.kinetics_model import KineticsModel
from hmfit.kinetics.simulate.synth import generate_dataset


def test_arrhenius_global_fit_recovers_parameters() -> None:
    mechanism_text = """
    species: A, B
    A -> B ; k1
    arrhenius: k1(A=A1, Ea=Ea1)
    """
    params_true = {"A1": 1500.0, "Ea1": 20000.0}
    params0 = {"A1": 1000.0, "Ea1": 15000.0}

    t = np.linspace(0.0, 4.0, 40)
    y0 = {"A": 1.0, "B": 0.0}
    rng = np.random.default_rng(3)
    A_true = rng.random((2, 3))
    temperatures = [290.0, 310.0, 330.0]

    datasets = generate_dataset(
        mechanism_text,
        params_true,
        A_true,
        t,
        y0,
        noise_level=1e-4,
        temperatures=temperatures,
    )

    model = KineticsModel(parse_mechanism(mechanism_text))
    objective = GlobalKineticsObjective(model, datasets)
    result = fit_global(
        objective,
        params0,
        bounds=(0.0, np.inf),
        max_nfev=300,
    )

    assert abs(result.params["Ea1"] - params_true["Ea1"]) / params_true["Ea1"] < 0.2
    assert abs(result.params["A1"] - params_true["A1"]) / params_true["A1"] < 0.3
