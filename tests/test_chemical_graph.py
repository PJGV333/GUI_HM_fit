import numpy as np

from hmfit_core.graph.chemical_graph import (
    ChemicalGraph,
    ReactionEdge,
    SpeciesNode,
    create_solver_inputs_from_graph,
)


def test_reaction_edge_balance_properties():
    h = SpeciesNode(name="H", charge=1)
    l = SpeciesNode(name="L", charge=-1)
    hl = SpeciesNode(name="HL", charge=0)
    edge = ReactionEdge(
        id="R1",
        reactants={h: 1.0, l: 1.0},
        products={hl: 1.0},
        log_beta=3.2,
    )

    assert edge.is_charge_balanced
    assert edge.is_mass_balanced


def test_reaction_edge_detects_mass_imbalance():
    m = SpeciesNode(name="M")
    l = SpeciesNode(name="L")
    ml2 = SpeciesNode(name="ML2")
    edge = ReactionEdge(
        id="R1",
        reactants={m: 1.0, l: 1.0},
        products={ml2: 1.0},
        log_beta=2.0,
    )

    assert not edge.is_mass_balanced
    assert edge.mass_balance_delta == {"L": 1.0}


def test_add_reaction_from_string_and_stoichiometric_matrix():
    graph = ChemicalGraph()
    graph.add_reaction_from_string("M + 2 L <=> ML2", log_beta=5.0)

    species, matrix = graph.to_stoichiometric_matrix()
    assert [node.name for node in species] == ["L", "M", "ML2"]
    assert matrix.shape == (1, 3)
    assert np.allclose(matrix, np.array([[-2.0, -1.0, 1.0]]))


def test_get_active_species_ignores_solids():
    graph = ChemicalGraph()
    graph.add_species(SpeciesNode(name="M", initial_concentration=1.0))
    graph.add_species(SpeciesNode(name="L", initial_concentration=2.0))
    graph.add_species(SpeciesNode(name="ML2", initial_concentration=0.0))
    graph.add_species(SpeciesNode(name="PPT", is_solid=True))
    graph.add_reaction_from_string("M + 2 L <=> ML2", log_beta=5.0)

    active = graph.get_active_species()
    assert [node.name for node in active] == ["M", "L", "ML2"]


def test_create_solver_inputs_from_graph_separates_components_and_complexes():
    graph = ChemicalGraph()
    graph.add_species(SpeciesNode(name="M", initial_concentration=1.0))
    graph.add_species(SpeciesNode(name="L", initial_concentration=2.0))
    graph.add_reaction_from_string("M + 2 L <=> ML2", log_beta=5.0)

    payload = create_solver_inputs_from_graph(graph)
    solver_inputs = payload["solver_inputs"]

    assert payload["components"] == ["L", "M"]
    assert payload["complexes"] == ["ML2"]
    assert np.allclose(payload["edge_log_beta"], np.array([5.0]))

    model = solver_inputs["modelo"]
    assert model.shape == (2, 3)
    assert np.allclose(model, np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]))
    assert np.allclose(solver_inputs["ctot"], np.array([[2.0, 1.0]]))
    assert np.allclose(solver_inputs["k"], np.array([5.0]))


def test_create_solver_inputs_from_graph_is_order_invariant():
    reactions = [
        ("H + C <=> HC", 3.0),
        ("H + A <=> HA", 4.0),
        ("HC + A <=> HCA", 4.5),
    ]

    g1 = ChemicalGraph()
    for reaction, log_beta in reactions:
        g1.add_reaction_from_string(reaction, log_beta=log_beta)

    g2 = ChemicalGraph()
    for reaction, log_beta in reversed(reactions):
        g2.add_reaction_from_string(reaction, log_beta=log_beta)

    p1 = create_solver_inputs_from_graph(g1)
    p2 = create_solver_inputs_from_graph(g2)

    assert p1["components"] == p2["components"] == ["A", "C", "H"]
    assert p1["complexes"] == p2["complexes"] == ["HA", "HC", "HCA"]
    assert np.allclose(p1["stoichiometric_matrix"], p2["stoichiometric_matrix"])
    assert np.allclose(p1["edge_log_beta"], p2["edge_log_beta"])


def test_pathway_resolution_flattening():
    graph = ChemicalGraph()
    graph.add_reaction_from_string("H + C <=> HC", log_beta=3.0)
    graph.add_reaction_from_string("HC + A <=> HCA", log_beta=4.0)

    global_stoich, global_log_beta = graph.resolve_global_pathways()

    assert np.isclose(global_log_beta["HCA"], 7.0)
    assert global_stoich["HCA"] == {"A": 1.0, "C": 1.0, "H": 1.0}

    payload = create_solver_inputs_from_graph(graph)
    components = payload["components"]
    complexes = payload["complexes"]
    model = np.asarray(payload["solver_inputs"]["modelo"], dtype=float)
    k_vals = np.asarray(payload["solver_inputs"]["k"], dtype=float)

    assert components == ["A", "C", "H"]
    assert complexes == ["HC", "HCA"]
    assert model.shape == (3, 5)

    expected_model = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(model, expected_model)
    assert np.allclose(k_vals, np.array([3.0, 7.0]))
