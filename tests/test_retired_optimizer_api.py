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
