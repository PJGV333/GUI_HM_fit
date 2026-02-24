import pytest

from hmfit_core.graph.chemical_graph import ChemicalGraph
from hmfit_core.graph.graph_solver_bridge import calculate_free_species
from hmfit_core.graph.phase_manager import update_phase_states


@pytest.fixture
def setup_heterotopic_system() -> ChemicalGraph:
    graph = ChemicalGraph()
    graph.add_reaction_from_string("H + C <=> HC", log_beta=3.0)
    graph.add_reaction_from_string("H + A <=> HA", log_beta=4.0)
    graph.add_reaction_from_string("HC + A <=> HCA", log_beta=4.5)
    graph.add_reaction_from_string("HA + C <=> HCA", log_beta=3.5)
    return graph


def test_thermodynamic_cycle_consistency(
    setup_heterotopic_system: ChemicalGraph,
) -> None:
    setup_heterotopic_system.validate_thermodynamic_cycles()


def test_thermodynamic_cycle_inconsistency() -> None:
    graph = ChemicalGraph()
    graph.add_reaction_from_string("H + C <=> HC", log_beta=3.0)
    graph.add_reaction_from_string("H + A <=> HA", log_beta=4.0)
    graph.add_reaction_from_string("HC + A <=> HCA", log_beta=4.5)
    graph.add_reaction_from_string("HA + C <=> HCA", log_beta=5.0)

    with pytest.raises(ValueError):
        graph.validate_thermodynamic_cycles()


def test_multicomponent_solver(setup_heterotopic_system: ChemicalGraph) -> None:
    total_conc = {"H": 0.01, "C": 0.02, "A": 0.02}
    concentrations = calculate_free_species(setup_heterotopic_system, total_conc)

    expected_keys = {"H", "C", "A", "HC", "HA", "HCA"}
    assert expected_keys.issubset(concentrations.keys())

    assert (
        concentrations["H"]
        + concentrations["HC"]
        + concentrations["HA"]
        + concentrations["HCA"]
    ) == pytest.approx(0.01)


def test_dynamic_precipitation() -> None:
    graph = ChemicalGraph()
    graph.add_reaction_from_string("M + L <=> ML", log_beta=5.0)

    current_conc = {"M": 1e-3, "L": 1e-3, "ML": 0.1}
    kps_limits = {"ML": 0.05}

    update_phase_states(graph, current_conc, kps_limits)

    node_ml = graph.get_species("ML")
    assert node_ml is not None
    assert node_ml.is_solid is True

    active_names = [species.name for species in graph.get_active_species()]
    assert "ML" not in active_names
