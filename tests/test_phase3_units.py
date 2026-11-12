"""Global fits must not silently combine incompatible kinetic units."""
import numpy as np
import pytest

from hmfit.kinetics.data.fit_dataset import KineticsFitDataset
from hmfit.kinetics.data.units import validate_dataset_units


def dataset(time="s", conc="M"):
    return KineticsFitDataset(t=np.array([0., 1.]), y0={"A": 1.}, D=np.ones((2, 1)),
                              time_unit=time, concentration_unit=conc)


def test_aliases_and_incompatible_units():
    assert not validate_dataset_units([dataset("seconds", "mol/L"), dataset("s", "M")])
    with pytest.raises(ValueError, match="time units"):
        validate_dataset_units([dataset("min"), dataset("s")])
    with pytest.raises(ValueError, match="concentration units"):
        validate_dataset_units([dataset(conc="mM"), dataset(conc="M")])
    assert len(validate_dataset_units([dataset("?", ""), dataset("s", "M")])) == 2


def test_mixed_units_are_rejected_before_fit():
    pytest.importorskip("PySide6.QtWidgets")
    from hmfit_gui_qt.tabs.kinetics_tab import _run_kinetics_fit
    from hmfit.kinetics.model.kinetics_model import KineticsModel
    from hmfit.kinetics.mechanism_editor.parser import parse_mechanism
    model = KineticsModel(parse_mechanism("species: A, B\nA -> B ; k\n"))
    t = np.linspace(0, 120, 25)
    y = {"A": 1., "B": 0.}
    d = model.solve_concentrations(t, y, {"k": .025}) @ np.array([[.1, .4], [1., .2]])
    datasets = [KineticsFitDataset(t=t, y0=y, D=d, time_unit="s"),
                KineticsFitDataset(t=t / 60, y0=y, D=d, time_unit="min")]
    cfg = {"model": model, "datasets": datasets, "params0": [.02], "bounds": ([0.], [np.inf]),
           "param_names": ["k"], "log_params": [], "fixed_mask": [False]}
    with pytest.raises(ValueError, match="incompatible time units"):
        _run_kinetics_fit(cfg)
