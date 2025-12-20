from hmfit_gui_qt.widgets.channel_spec import DEFAULT_CHANNEL_TOL, parse_channel_spec


def test_parse_channel_spec_range():
    available = [300.0, 350.0, 400.0, 485.0, 510.0, 520.0, 550.0, 600.0]
    result = parse_channel_spec("350-550", available, tol=DEFAULT_CHANNEL_TOL)
    assert result.mode == "custom"
    assert result.errors == []
    assert result == {350.0, 400.0, 485.0, 510.0, 520.0, 550.0}


def test_parse_channel_spec_all():
    available = [300.0, 350.0, 400.0]
    result = parse_channel_spec("all", available, tol=DEFAULT_CHANNEL_TOL)
    assert result.mode == "all"
    assert result.errors == []
    assert result == {300.0, 350.0, 400.0}


def test_parse_channel_spec_values_with_nearest():
    available = [349.8, 401.0, 550.3]
    result = parse_channel_spec("350, 550", available, tol=DEFAULT_CHANNEL_TOL)
    assert result.mode == "custom"
    assert result.errors == []
    assert result == {349.8, 550.3}


def test_parse_channel_spec_mixed():
    available = [300.0, 350.0, 400.0, 485.0, 510.0, 520.0, 550.0]
    result = parse_channel_spec("350-550, 485", available, tol=DEFAULT_CHANNEL_TOL)
    assert result.mode == "custom"
    assert result.errors == []
    assert result == {350.0, 400.0, 485.0, 510.0, 520.0, 550.0}
