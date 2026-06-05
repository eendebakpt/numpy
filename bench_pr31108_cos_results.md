# Benchmark Results: PR #31108 vs merge base (bm_numpy_cos.py)

PR: https://github.com/numpy/numpy/pull/31108
Title: ENH, PERF: small arrays/dimensions on array object - with special allocator
Benchmark script: ~/python_testing/benchmarks/bm_numpy_cos.py
Date: 2026-04-04
Python: 3.13.3 (release build, -O2 -DNDEBUG, GCC 14.2.0)
NumPy: 2.5.0.dev0
Base commit: 6de80d1e4e (merge base)
PR commit: d82dbeaba1

| Benchmark | merge base | PR #31108 | Change |
|---|---|---|---|
| `np.abs(x)` | 323 ns | 260 ns | 1.24x faster |
| `np.cos(x)` | 318 ns | 258 ns | 1.23x faster |
| `np.add.accumulate(x)` | 412 ns | 352 ns | 1.17x faster |
| `np.add.reduce(x)` | 724 ns | 669 ns | 1.08x faster |
| `np.abs(-2.2)` | 83.2 ns | 87.1 ns | 1.05x slower |
| `np.cos(-2.2)` | 74.1 ns | 77.1 ns | 1.04x slower |
| `np.abs(np.float64(-2.2))` | 120 ns | 123 ns | 1.02x slower |
| `np.cos(np.float64(-2.2))` | 111 ns | 113 ns | 1.01x slower |
| Geometric mean | (ref) | 1.07x faster | |
