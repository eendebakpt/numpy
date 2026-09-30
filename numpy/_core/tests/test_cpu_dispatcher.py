import pytest

from numpy._core import _umath_tests
from numpy._core._multiarray_umath import (
    __cpu_baseline__,
    __cpu_dispatch__,
    __cpu_features__,
)
from numpy.testing import assert_equal


def test_dispatcher():
    """
    Testing the utilities of the CPU dispatcher
    """
    targets = (
        "X86_V2", "X86_V3",
        "VSX", "VSX2", "VSX3",
        "NEON", "ASIMD", "ASIMDHP",
        "VX", "VXE", "LSX", "RVV"
    )
    highest_sfx = ""  # no suffix for the baseline
    all_sfx = []
    for feature in reversed(targets):
        # skip baseline features, by the default `CCompilerOpt` do not generate
        # separated objects for the baseline, just one object combined all of them
        # via 'baseline' option within the configuration statements.
        if feature in __cpu_baseline__:
            continue
        # check compiler and running machine support
        if feature not in __cpu_dispatch__ or not __cpu_features__[feature]:
            continue

        if not highest_sfx:
            highest_sfx = "_" + feature
        all_sfx.append("func" + "_" + feature)

    test = _umath_tests.test_dispatch()
    assert_equal(test["func"], "func" + highest_sfx)
    assert_equal(test["var"], "var" + highest_sfx)

    if highest_sfx:
        assert_equal(test["func_xb"], "func" + highest_sfx)
        assert_equal(test["var_xb"], "var" + highest_sfx)
    else:
        assert_equal(test["func_xb"], "nobase")
        assert_equal(test["var_xb"], "nobase")

    all_sfx.append("func")  # add the baseline
    assert_equal(test["all"], all_sfx)


def test_lazy_module_attributes():
    # `__cpu_targets_info__` and `__cpu_features__` are built on first access
    import numpy._core._multiarray_umath as mu
    from numpy._core._multiarray_umath import __cpu_targets_info__

    assert __cpu_targets_info__ is mu.__cpu_targets_info__
    assert __cpu_targets_info__ is mu.__dict__["__cpu_targets_info__"]
    assert set(__cpu_targets_info__["add"]["ddd"]) == {"current", "available"}
    assert isinstance(mu.__cpu_features__, dict)
    with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
        mu.nonexistent
