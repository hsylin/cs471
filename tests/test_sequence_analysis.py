import pytest
from sequence_study.analysis import geometric_nonnegative

def test_valid_zero_and_missing_drain():
    assert geometric_nonnegative([0.,2.])==0.
    assert geometric_nonnegative([None,2.]) is None
    assert geometric_nonnegative([.5,2.])==1.

def test_invalid_ratio_rejected():
    for values in ([-1.,2.],[float('nan'),2.]):
        with pytest.raises(ValueError):geometric_nonnegative(values)
