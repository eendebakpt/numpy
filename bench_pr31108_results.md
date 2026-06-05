# Benchmark Results: PR #31108 vs main

PR: https://github.com/numpy/numpy/pull/31108
Title: ENH, PERF: small arrays/dimensions on array object - with special allocator
Benchmark script: bench_ufunc_performance.py
Date: 2026-04-01
Python: 3.13.3 (release build, GCC 14.2.0)
NumPy: 2.5.0.dev0

| Benchmark | main | PR #31108 | Change |
|---|---|---|---|
| `np.abs(array_1)` | 312 ns | 241 ns | **1.29x faster** |
| `np.sin(array_1)` | 304 ns | 236 ns | **1.29x faster** |
| `np.abs(array_4)` | 314 ns | 242 ns | **1.30x faster** |
| `np.sin(array_4)` | 323 ns | 254 ns | **1.27x faster** |
| `np.abs(array_100)` | 320 ns | 306 ns | 1.05x faster |
| `np.sin(array_100)` | 810 ns | 800 ns | 1.01x faster |
| `np.sin(array_2x2)` | 325 ns | 254 ns | **1.28x faster** |
| `np.sin(int_array_4)` | 563 ns | 432 ns | **1.30x faster** |
| `np.sum(array_100)` | 1.52 us | 1.47 us | 1.04x faster |
| `np.float64(1.0)` | 43.6 ns | 41.9 ns | 1.04x faster |
| `np.sin(python_float)` | 66.8 ns | 67.3 ns | 1.01x slower |
| `np.abs(float64)` | 101 ns | 111 ns | 1.10x slower |
| `np.sin(float64)` | 90.0 ns | 105 ns | 1.16x slower |
| **Geometric mean** | (ref) | **1.11x faster** | |
