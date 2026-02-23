import numpy as np
import pytest

from hmfit_core.graph.chemical_graph import ChemicalGraph, SpeciesNode
from hmfit_core.graph.graph_solver_bridge import calculate_free_species
from hmfit_core.graph.phase_manager import update_phase_states


def test_validate_thermodynamic_cycles_passes_for_consistent_network():
    graph = ChemicalGraph()
    graph.add_reaction_from_string("A <=> B", log_beta=1.0)
    graph.add_reaction_from_string("B <=> C", log_beta=2.0)
    graph.add_reaction_from_string("A <=> C", log_beta=3.0)

    graph.validate_thermodynamic_cycles()


def test_validate_thermodynamic_cycles_raises_for_inconsistent_network():
    graph = ChemicalGraph()
    graph.add_reaction_from_string("A <=> B", log_beta=1.0)
    graph.add_reaction_from_string("B <=> C", log_beta=2.0)
    graph.add_reaction_from_string("A <=> C", log_beta=2.5)

    with pytest.raises(ValueError, match="Thermodynamic inconsistency"):
        graph.validate_thermodynamic_cycles()


def test_calculate_free_species_bridge_with_nr_solver():
    graph = ChemicalGraph()
    graph.add_species(SpeciesNode(name="M"))
    graph.add_species(SpeciesNode(name="L"))
    graph.add_reaction_from_string("M + 2 L <=> ML2", log_beta=5.0)

    totals = {"M": 1.0e-3, "L": 2.0e-3}
    concentrations = calculate_free_species(graph=graph, total_concentrations=totals)

    assert set(concentrations.keys()) == {"M", "L", "ML2"}
    assert all(value >= 0.0 for value in concentrations.values())

    assert np.isclose(
        concentrations["M"] + concentrations["ML2"],
        totals["M"],
        rtol=1e-6,
        atol=1e-10,
    )
    assert np.isclose(
        concentrations["L"] + 2.0 * concentrations["ML2"],
        totals["L"],
        rtol=1e-6,
        atol=1e-10,
    )


def test_update_phase_states_marks_species_as_solid():
    graph = ChemicalGraph()
    graph.add_species(SpeciesNode(name="Na"))
    graph.add_species(SpeciesNode(name="Cl"))
    graph.add_species(SpeciesNode(name="NaCl"))
    graph.add_reaction_from_string("Na + Cl <=> NaCl", log_beta=0.0)

    first = update_phase_states(
        graph=graph,
        current_concentrations={"Na": 1.0e-3, "Cl": 1.0e-3},
        solubility_products={"NaCl": 1.0e-4},
    )
    assert first == {"NaCl": False}
    nacl = graph.get_species("NaCl")
    assert nacl is not None
    assert not nacl.is_solid

    second = update_phase_states(
        graph=graph,
        current_concentrations={"Na": 1.0e-1, "Cl": 1.0e-1},
        solubility_products={"NaCl": 1.0e-4},
    )
    assert second == {"NaCl": True}
    nacl = graph.get_species("NaCl")
    assert nacl is not None
    assert nacl.is_solid

    active_species, _ = graph.to_stoichiometric_matrix()
    assert [node.name for node in active_species] == ["Na", "Cl"]
