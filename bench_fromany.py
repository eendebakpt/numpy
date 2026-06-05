"""Benchmark PyArray_FromAny fast path impact.

Uses small arrays to maximize the relative impact of the overhead reduction.
"""
import pyperf
import numpy as np

runner = pyperf.Runner()

a4 = np.ones(4)
a4x4 = np.ones((4, 4))

# Reductions (call PyArray_FromAny(op, NULL, 0, 0, 0, NULL))
runner.bench_func('np.sum(array_4)', np.sum, a4)
runner.bench_func('np.prod(array_4)', np.prod, a4)
runner.bench_func('np.max(array_4)', np.max, a4)
runner.bench_func('np.any(bool_4)', np.any, np.ones(4, dtype=bool))
runner.bench_func('np.sum(4x4)', np.sum, a4x4)

# copyto (calls PyArray_FromAny on src)
dst = np.empty(4)
runner.bench_func('np.copyto(dst,src)', np.copyto, dst, a4)

# Ufunc with out= (calls PyArray_FromAny on out arg)
out = np.empty(4)
runner.bench_func('np.sin(a,out=out)', np.sin, a4, out)

# Unary ufunc without out= (does NOT call PyArray_FromAny)
runner.bench_func('np.sin(array_4)', np.sin, a4)

# Operations that do NOT hit fast path
runner.bench_func('np.count_nonzero(4)', np.count_nonzero, a4)

# np.where (hits fast path on all 3 input arrays)
a_bool = np.ones(4, dtype=bool)
b = np.zeros(4)
runner.bench_func('np.where(cond,a,b)', np.where, a_bool, a4, b)
