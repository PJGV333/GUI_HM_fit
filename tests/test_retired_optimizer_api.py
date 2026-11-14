"""Public fit API rejects retired optimizers without implicit substitution."""
from dataclasses import dataclass

import pytest

from hmfit_core.api import run_nmr_fit, run_spectroscopy_fit


@dataclass
class OldConfig:
    optimizer: str = "explicit_lm"


@pytest.mark.parametrize("fit", [run_nmr_fit, run_spectroscopy_fit])
@pytest.mark.parametrize("config", [{"optimizer": "explicit_lm"}, OldConfig()])
def test_retired_optimizer_rejected_before_processing(fit, config):
    messages = []
    with pytest.raises(ValueError, match="(?i)explicit_lm.*powell"):
        fit(config, progress_cb=messages.append)
    assert messages == []


@pytest.mark.parametrize("config", [{"optimizer": "explicit_lm"}, OldConfig(),
                                  {"optimizer": " EXPLICIT_LM "}])
def test_direct_nmr_runner_rejects_retired_optimizer_before_mutating_callback(monkeypatch, config):
    from hmfit_core.processors import nmr_processor
    from hmfit_core.run_nmr import run_nmr

    previous = lambda _message: None
    monkeypatch.setattr(nmr_processor, "_progress_callback", previous)
    messages = []
    with pytest.raises(ValueError, match="(?i)explicit_lm.*powell"):
        run_nmr(config, progress_cb=messages.append)
    assert messages == []
    assert nmr_processor._progress_callback is previous


@pytest.mark.parametrize("config", [{"optimizer": "explicit_lm"}, OldConfig(),
                                  {"optimizer": " EXPLICIT_LM "}])
def test_direct_spectroscopy_runner_rejects_retired_optimizer_before_mutating_callback(monkeypatch, config):
    from hmfit_core.processors import spectroscopy_processor
    from hmfit_core.run_spectroscopy import run_spectroscopy

    previous = lambda _message: None
    monkeypatch.setattr(spectroscopy_processor, "_progress_callback", previous)
    messages = []
    with pytest.raises(ValueError, match="(?i)explicit_lm.*powell"):
        run_spectroscopy(config, progress_cb=messages.append)
    assert messages == []
    assert spectroscopy_processor._progress_callback is previous
