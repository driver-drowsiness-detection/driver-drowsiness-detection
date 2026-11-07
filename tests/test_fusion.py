from dataclasses import replace
import math
import pytest
from marvel.contracts import Estimate, CLOSED, UNKNOWN
from marvel.cognition.fusion import combine, score_state

def evidence(method, value):
    return Estimate("frame", 1, "driver", "left", method, value, value is not None)

@pytest.mark.parametrize("alpha,expected", [(0, .7), (.5, .8), (1, .9)])
def test_weight_direction_and_endpoints(alpha, expected):
    result = combine(evidence("ear_probability", .7), evidence("cnn", .9), alpha=alpha)
    assert result.value == pytest.approx(expected)
    assert result.method == "fusion" and result.sample_id == "frame"

@pytest.mark.parametrize("alpha", [0, .5, 1])
@pytest.mark.parametrize("missing", ["ear_probability", "cnn"])
def test_missing_is_not_zero_even_at_endpoints(alpha, missing):
    result = combine(evidence("ear_probability", None if missing == "ear_probability" else .7),
                     evidence("cnn", None if missing == "cnn" else .9), alpha=alpha)
    assert result.value is None and not result.valid
    assert score_state(result, .5) == UNKNOWN

@pytest.mark.parametrize("field,value", [("sample_id", "other"), ("t", 2),
                                        ("track", "other"), ("side", "right")])
def test_mixed_identities_rejected(field, value):
    with pytest.raises(ValueError, match="different"):
        combine(evidence("ear_probability", .7),
                replace(evidence("cnn", .9), **{field: value}), alpha=.5)

@pytest.mark.parametrize("value", [-.1, 1.1, math.nan, math.inf])
def test_invalid_parameters(value):
    with pytest.raises(ValueError):
        combine(evidence("ear_probability", .7), evidence("cnn", .9), alpha=value)
    with pytest.raises(ValueError):
        score_state(evidence("cnn", .9), value)

def test_raw_ear_and_invalid_score_rejected():
    with pytest.raises(ValueError):
        combine(evidence("ear", .2), evidence("cnn", .9), alpha=.5)
    with pytest.raises(ValueError):
        combine(evidence("ear_probability", .7), evidence("cnn", 2), alpha=.5)
    assert score_state(evidence("cnn", .5), .5) == CLOSED
