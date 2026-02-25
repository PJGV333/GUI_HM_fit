import pytest

from hmfit_core.utils.graph_gui_parser import parse_multiline_equilibria


def test_parse_multiline_equilibria_valid_block():
    text = """
    # comment
    H + C <=> HC ; 3.0
    H + A <=> HA ; logB=4.0
    HC + A <=> HCA ; log_beta = 4.5
    """

    graph, payload = parse_multiline_equilibria(text)

    assert len(graph.reactions) == 3
    assert payload["components"] == ["H", "C", "A"]
    assert payload["complexes"] == ["HC", "HA", "HCA"]


def test_parse_multiline_equilibria_reports_line_errors():
    text = """
    H + C <=> HC ; 3.0
    bad line without separator
    """
    with pytest.raises(ValueError, match="Line 3"):
        parse_multiline_equilibria(text)


def test_parse_non_absorbing_tags():
    text = """
    H + C <=> HC ; 3.0 @NA C
    H + A <=> HA ; 4.0 @na A
    HC + A <=> HCA ; 4.5 @Na HCA, HA
    """

    _, solver_inputs = parse_multiline_equilibria(text)

    assert "non_abs_species" in solver_inputs
    assert set(solver_inputs["non_abs_species"]) == {"C", "A", "HA", "HCA"}


def test_parse_mixed_na_declarations():
    text = """
    R + Q <=> RQ ; 3.0 @NA Q
    RQ + X <=> RQX ; 4.0
    @na X, Solvente
    """

    _, solver_inputs = parse_multiline_equilibria(text)

    assert "non_abs_species" in solver_inputs
    assert set(solver_inputs["non_abs_species"]) == {"Q", "X", "Solvente"}


def test_parse_inline_na_without_names_marks_products():
    text = """
    r + q + x <=> rqx ; 5 @na
    r + x <=> rx ; 4
    @na x
    """

    _, solver_inputs = parse_multiline_equilibria(text)

    assert "non_abs_species" in solver_inputs
    assert set(solver_inputs["non_abs_species"]) == {"rqx", "x"}
