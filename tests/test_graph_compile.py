import numpy as np

from hmfit_core.graph.hmgraph import HMGraph


def test_compile_simple_system():
    g = HMGraph()
    g.add_component("H")
    g.add_component("G")

    g.add_species("HG", stoich={"H": 1, "G": 1})
    g.add_species("H2G", stoich={"H": 2, "G": 1})

    # redundante, pero asegura aristas explicitas
    g.connect("H", "HG", 1)
    g.connect("G", "HG", 1)
    g.connect("H", "H2G", 2)
    g.connect("G", "H2G", 1)

    mtx, nas = g.compile()
    assert nas.size == 0

    # Orden esperado: [H, G, HG, H2G]
    expected = np.array(
        [
            [1, 0, 1, 2],
            [0, 1, 1, 1],
        ],
        dtype=float,
    )
    assert mtx.shape == expected.shape
    assert np.allclose(mtx, expected)


def test_compile_with_inactive_species():
    g = HMGraph()
    g.add_component("H")
    g.add_component("G")
    g.add_species("HG", stoich={"H": 1, "G": 1})
    g.add_species("H2G", stoich={"H": 2, "G": 1})

    g.set_active("H2G", False)
    mtx, nas = g.compile()

    expected = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    assert np.allclose(mtx, expected)
    assert nas.size == 0


def test_compile_nas_observable_flag():
    g = HMGraph()
    g.add_component("H")
    g.add_component("G", observable=False)  # G no observable
    g.add_species("HG", stoich={"H": 1, "G": 1}, observable=True)

    mtx, nas = g.compile()
    assert mtx.shape == (2, 3)
    assert nas.tolist() == [1]
