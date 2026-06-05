"""Benchmark script for reduction performance improvements.

Measures performance of reductions (sum, prod, max, min, any) for:
- Small 1D arrays (size 4, 20)
- Int array (size 20)
- 2D array (10x10, full reduction and axis reduction)
- Array subclass (size 20) to exercise the __array_function__ dispatch path
- Comparison with non-reduction ufunc (sin) as a baseline
"""
import pyperf
import numpy as np


class _Sub(np.ndarray):
    """Trivial ndarray subclass to trigger the __array_function__ path."""


runner = pyperf.Runner()

# 1D float reductions
a4 = np.ones(4)
a20 = np.ones(20)
runner.bench_func('np.sum(array_4)', np.sum, a4)
runner.bench_func('np.sum(array_20)', np.sum, a20)
runner.bench_func('np.prod(array_20)', np.prod, a20)
runner.bench_func('np.max(array_20)', np.max, a20)

# 1D int reduction
a_int = np.ones(20, dtype=np.int64)
runner.bench_func('np.sum(int_20)', np.sum, a_int)

# 2D reductions
a2d = np.ones((10, 10))
runner.bench_func('np.sum(10x10)', np.sum, a2d)
runner.bench_func('np.sum(10x10,axis=0)', np.sum, a2d, 0)
runner.bench_func('np.sum(10x10,axis=1)', np.sum, a2d, 1)

# Boolean reduction
a_bool = np.ones(20, dtype=bool)
runner.bench_func('np.any(bool_20)', np.any, a_bool)

# Small 2D reductions (full)
a2x2 = np.ones((2, 2))
a2x2_int = np.ones((2, 2), dtype=np.int64)
runner.bench_func('np.sum(2x2_float)', np.sum, a2x2)
runner.bench_func('np.sum(2x2_int)', np.sum, a2x2_int)

# Subclass: exercises the __array_function__ dispatch path
a_sub = np.ones(20).view(_Sub)
runner.bench_func('np.sum(array_subclass_20)', np.sum, a_sub)

# Reference: unary ufunc (not a reduction, does not go through
# __array_function__ -- performance should be unchanged by this PR)
runner.bench_func('np.sin(array_20)', np.sin, a20)
