"""Integration tests for adapting KineticGraph into KineticsModel."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from hmfit.kinetics.model.graph_adapter import build_kinetics_model_from_graph
from hmfit.kinetics.model.kinetics_graph import KineticGraph
from hmfit.kinetics.model.kinetics_model import KineticsModel
from hmfit.kinetics.model.reactions import Reaction


def test_graph_to_kinetics_model() -> None:
    graph = KineticGraph()
    graph.add_kinetic_reaction("A -> B", k_forward=2.0)
    graph.add_kinetic_reaction("B -> C", k_forward=1.0)

    model = build_kinetics_model_from_graph(graph)

    assert isinstance(model, KineticsModel)
    assert len(model.reactions) == 2
    assert all(isinstance(reaction, Reaction) for reaction in model.reactions)

    first, second = model.reactions
    assert first.reactants == {"A": 1}
    assert first.products == {"B": 1}
    assert second.reactants == {"B": 1}
    assert second.products == {"C": 1}
