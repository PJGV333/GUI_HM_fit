import pytest

from hmfit_core.utils.graph_gui_parser import parse_multiline_equilibria


def test_parse_absgrp_directives_successfully():
    text = """
    rh2 + f <=> rh2f ; 5
    rh2f + f <=> rhm + hf2 ; 5
    @na f hf2
    @absgrp 1: rh2
    @absgrp 2: rh2f rhm
    """

    _, solver_inputs = parse_multiline_equilibria(text)

    assert solver_inputs["species_names"] == ["rh2", "f", "rh2f", "rhm", "hf2"]
    assert set(solver_inputs["non_abs_species"]) == {"f", "hf2"}
    assert solver_inputs["abs_groups"] == {"1": ["rh2"], "2": ["rh2f", "rhm"]}


def test_parse_absgrp_rejects_duplicate_species():
    text = """
    H + G <=> HG ; 4
    HG + G <=> HG2 ; 3
    @absgrp a: H HG
    @absgrp b: HG2 HG
    """

    with pytest.raises(ValueError, match="already assigned to absorptivity group"):
        parse_multiline_equilibria(text)


def test_parse_absgrp_rejects_unknown_species():
    text = """
    H + G <=> HG ; 4
    @absgrp 1: H HX
    """

    with pytest.raises(ValueError, match="Unknown species 'HX'"):
        parse_multiline_equilibria(text)


def test_parse_absgrp_rejects_non_absorbent_species():
    text = """
    H + G <=> HG ; 4
    @na HG
    @absgrp x: H HG
    """

    with pytest.raises(ValueError, match="marked as @na and cannot be used in @absgrp"):
        parse_multiline_equilibria(text)
