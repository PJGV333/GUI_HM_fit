import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
pytest.importorskip("numpy")
pytest.importorskip("pandas")
openpyxl = pytest.importorskip("openpyxl")


def test_dependency_imports_smoke() -> None:
    assert hasattr(matplotlib, "__version__")
    assert hasattr(openpyxl, "__version__")
