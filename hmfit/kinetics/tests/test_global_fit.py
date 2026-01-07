"""Tests for global kinetics fitting."""

from __future__ import annotations

import numpy as np

from hmfit.kinetics.fit.objective import GlobalKineticsObjective
from hmfit.kinetics.fit.optimizer import fit_global
from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
from hmfit.kinetics.model.kinetics_model import KineticsModel
from hmfit.kinetics.simulate.synth import generate_dataset


def test_global_fit_recovers_rates() -> None:
    mechanism_text = """
    species: A, B, C
    A -> B ; k1
    B -> C ; k2
    """
    params_true = {"k1": 1.2, "k2": 0.4}
    params0 = {"k1": 0.8, "k2": 0.2}

    t = np.linspace(0.0, 4.0, 60)
    y0 = {"A": 1.0, "B": 0.0, "C": 0.0}
    rng = np.random.default_rng(2)
    A_true = rng.random((3, 4))

    dataset = generate_dataset(
        mechanism_text,
        params_true,
        A_true,
        t,
        y0,
        noise_level=1e-4,
        temperatures=298.15,
    )

    model = KineticsModel(parse_mechanism(mechanism_text))
    objective = GlobalKineticsObjective(model, [dataset])
    result = fit_global(
        objective,
        params0,
        bounds=(0.0, np.inf),
        max_nfev=200,
    )

    assert abs(result.params["k1"] - params_true["k1"]) / params_true["k1"] < 0.1
    assert abs(result.params["k2"] - params_true["k2"]) / params_true["k2"] < 0.1
