"""Benchmark script for ufunc performance improvements.

Measures performance of ufuncs (sin, abs) for:
- Small numpy arrays (size 1, 4, 100)
- np.float64 scalar
- Python float
"""
import pyperf
import numpy as np

runner = pyperf.Runner()

# Python float input
runner.bench_func('np.abs(python_float)', np.abs, -2.2)
runner.bench_func('np.sin(python_float)', np.sin, 1.0)

# np.float64 scalar input
x64 = np.float64(-2.2)
runner.bench_func('np.abs(float64)', np.abs, x64)
runner.bench_func('np.sin(float64)', np.sin, np.float64(1.0))

# np.float64 scalar creation
runner.bench_func('np.float64(1.0)', np.float64, 1.0)

# Small arrays
a1 = np.ones(1)
a4 = np.ones(4)
a100 = np.ones(100)

runner.bench_func('np.abs(array_1)', np.abs, a1)
runner.bench_func('np.sin(array_1)', np.sin, a1)
runner.bench_func('np.abs(array_4)', np.abs, a4)
runner.bench_func('np.sin(array_4)', np.sin, a4)
runner.bench_func('np.abs(array_100)', np.abs, a100)
runner.bench_func('np.sin(array_100)', np.sin, a100)

# 2D array
a2x2 = np.ones((2, 2))
runner.bench_func('np.sin(array_2x2)', np.sin, a2x2)

# Int array (requires cast)
a_int = np.ones(4, dtype=np.int64)
runner.bench_func('np.sin(int_array_4)', np.sin, a_int)

# Reduction
runner.bench_func('np.sum(array_100)', np.sum, a100)
