# PyArray_FromAny Fast Path Benchmark Results

PR: https://github.com/numpy/numpy/pull/31099
Branch: `pyarray_fromany_fast_path` (commit 6fc2db4d66) vs `main`
Python: 3.13.3 (release build, -O2 -DNDEBUG, GCC 14.2.0)
NumPy: 2.5.0.dev0
Platform: Linux x86_64
Date: 2026-04-10
Tool: pyperf

## Change

Add a fast path at the top of `PyArray_FromAny()`: when `op` is already
a `PyArrayObject` and no dtype, flags, depth, or context constraints are
given, return it directly with `Py_INCREF`. Skips the full
`DiscoverDTypeAndShape` -> `PyArray_CanCastArrayTo` -> `PyArray_FromArray`
pipeline (~680 instructions).

## Results: main vs branch

| Benchmark | main | branch | Change |
|---|---|---|---|
| `np.copyto(dst, src)` | 195 ns | 141 ns | 1.38x faster |
| `np.where(cond, a, b)` | 643 ns | 536 ns | 1.20x faster |
| `np.count_nonzero(4)` | 141 ns | 131 ns | 1.08x faster |
| `np.any(bool_4)` | 1.39 us | 1.33 us | 1.04x faster |
| `np.sum(array_4)` | 1.52 us | 1.47 us | 1.03x faster |
| `np.prod(array_4)` | 1.49 us | 1.44 us | 1.03x faster |
| `np.sum(4x4)` | 1.55 us | 1.51 us | 1.03x faster |
| `np.max(array_4)` | 1.53 us | 1.50 us | 1.02x faster |
| `np.sin(a, out=out)` | 258 ns | 254 ns | 1.01x faster |
| `np.sin(array_4)` | n/s | n/s | not significant |
| Geometric mean | (ref) | 1.08x faster | |

## Call sites that hit the fast path

| Location | Function | Frequency |
|---|---|---|
| `ufunc_object.c:3685` | Reduction input | Every np.sum/prod/max/min/any/all |
| `ufunc_object.c:643` | Ufunc output arg | Every ufunc with out= |
| `multiarraymodule.c:1974` | np.copyto src | Every np.copyto |
| `arraywrap.c:229` | Array wrapping | After ufunc results |
| `iterators.c:1249` | Flat iterator setitem | arr.flat[i] = v |

## Call sites that miss the fast path (14 sites)

All pass dtype, flags, min_depth, or context arguments:
einsum, correlate, inner, copyto wheremask, count_nonzero,
bincount, digitize, interp, take/put/choose/where (index args),
busday operations, partition, array __setstate__, ufunc.at,
reduceat indices.

Miss overhead: ~5 extra instructions (negligible).

## Benchmark script

See `bench_fromany.py`.
