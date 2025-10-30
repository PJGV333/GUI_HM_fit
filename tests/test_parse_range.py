import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils import parse_range


def test_parse_range_valid():
    start, end, samples = parse_range("1e-3,2e-5,50")
    assert math.isclose(start, 1e-3)
    assert math.isclose(end, 2e-5)
    assert math.isclose(samples, 50.0)


def test_parse_range_strips_spaces():
    start, end, samples = parse_range(" 1e-3 , 2e-5 , 50 ")
    assert math.isclose(start, 1e-3)
    assert math.isclose(end, 2e-5)
    assert math.isclose(samples, 50.0)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "1e-3,2e-5",
        "foo,1,2",
        "1e-3,2e-5,-5",
    ],
)
def test_parse_range_invalid(value):
    with pytest.raises(ValueError):
        parse_range(value)
