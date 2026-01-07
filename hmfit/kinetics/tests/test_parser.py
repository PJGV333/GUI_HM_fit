"""Tests for mechanism parser."""

from hmfit.kinetics.mechanism_editor.parser import parse_mechanism


def test_parse_reversible_and_irreversible() -> None:
    text = """
    # comment line
    species: A, B, C
    fixed: H2O
    A -> B ; k1
    B <-> C ; k2, k_2
    """
    ast = parse_mechanism(text)

    assert ast.species == ["A", "B", "C"]
    assert ast.fixed == {"H2O"}
    assert len(ast.reactions) == 2

    r0 = ast.reactions[0]
    assert r0.reactants == {"A": 1}
    assert r0.products == {"B": 1}
    assert r0.k_forward == "k1"
    assert r0.k_reverse is None

    r1 = ast.reactions[1]
    assert r1.reactants == {"B": 1}
    assert r1.products == {"C": 1}
    assert r1.k_forward == "k2"
    assert r1.k_reverse == "k_2"


def test_parse_coefficients_and_spacing() -> None:
    text = """
    species: A, B, C, D
    2A + B -> C ; k4
    A + C -> D ; k3
    """
    ast = parse_mechanism(text)

    assert ast.reactions[0].reactants == {"A": 2, "B": 1}
    assert ast.reactions[0].products == {"C": 1}


def test_parse_temperature_models_inline_and_block() -> None:
    text = """
    species: A, B
    A -> B ; k1
    arrhenius: k1(A=A1, Ea=Ea1)
    eyring:
      k2(dH=dH2, dS=dS2)
    """
    ast = parse_mechanism(text)

    assert ast.temp_models["k1"].kind == "arrhenius"
    assert ast.temp_models["k1"].params == {"A": "A1", "Ea": "Ea1"}
    assert ast.temp_models["k2"].kind == "eyring"
    assert ast.temp_models["k2"].params == {"dH": "dH2", "dS": "dS2"}
