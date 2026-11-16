"""Retired external LM must be rejected at the lowest processing entrypoints."""
import inspect

import pytest

from hmfit_core.processors import nmr_processor, spectroscopy_processor


@pytest.mark.parametrize('processor', [nmr_processor.process_nmr_data,
                                      spectroscopy_processor.process_spectroscopy_data])
@pytest.mark.parametrize('optimizer', ['explicit_lm', ' Explicit_LM '])
def test_processor_rejects_retired_optimizer_before_reading_data(processor, optimizer, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('input must not be read')
    monkeypatch.setattr(nmr_processor.pd, 'read_excel', unexpected)
    kwargs = {name: None for name, p in inspect.signature(processor).parameters.items()
              if p.default is inspect.Parameter.empty}
    kwargs['optimizer'] = optimizer
    with pytest.raises(ValueError, match=r'(?i)explicit_lm.*retired.*powell'):
        processor(**kwargs)
