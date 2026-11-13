"""Rate-order validation applies to the core fitting API without Qt."""
import numpy as np
import pytest

from hmfit.kinetics.data.fit_dataset import KineticsFitDataset
from hmfit.kinetics.fit.objective import GlobalKineticsObjective
from hmfit.kinetics.fit.optimizer import fit_global
from hmfit.kinetics.fit.rate_parameters import _shared_rate_order_error
from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
from hmfit.kinetics.model.kinetics_model import KineticsModel


def test_fit_global_rejects_incompatible_shared_rate_with_gui_message():
    model = KineticsModel(parse_mechanism(
        "species: A, B, C\nA -> B ; k\nA + B -> C ; k\n"
    ))
    dataset = KineticsFitDataset(t=np.array([0., 1.]), y0={"A": 1., "B": 0., "C": 0.},
                                 D=np.ones((2, 1)))
    objective = GlobalKineticsObjective(model, [dataset], param_names=["k"])
    with pytest.raises(ValueError) as exc:
        fit_global(objective, {"k": 1.})
    assert str(exc.value) == _shared_rate_order_error(model)
    assert "reactions of different orders" in str(exc.value)
