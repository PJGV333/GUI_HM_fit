import pytest
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from hmfit_core.utils.graph_gui_parser import (
    clear_parse_cache,
    get_parse_cache_stats,
    normalize_equilibria_text,
    parse_multiline_equilibria,
)


@pytest.fixture(autouse=True)
def _reset_parser_cache():
    clear_parse_cache()
    yield
    clear_parse_cache()


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


def test_parser_cache_hit_and_miss_for_equivalent_text():
    text_a = """
    H + C <=> HC ; 3.0
    H + A <=> HA ; logB=4.0
    """
    text_b = """
      H  +   C   <=> HC ; 3.0

      H + A <=> HA ; logB = 4.0
    """

    _, solver_a = parse_multiline_equilibria(text_a)
    _, solver_b = parse_multiline_equilibria(text_b)
    stats = get_parse_cache_stats()

    assert solver_a["components"] == solver_b["components"]
    assert solver_a["complexes"] == solver_b["complexes"]
    assert np.allclose(solver_a["stoichiometric_matrix"], solver_b["stoichiometric_matrix"])
    assert np.allclose(solver_a["edge_log_beta"], solver_b["edge_log_beta"])
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["size"] == 1


def test_normalize_equilibria_text_stabilizes_hash_input():
    raw = """
      H +   G  <=>   HG ; logB=4.5
      # comment
      @NA   HG
    """
    norm = normalize_equilibria_text(raw)
    assert "H + G <=> HG ; logB=4.5" in norm
    assert "@na HG" in norm
