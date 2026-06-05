# Reduction Performance Benchmarks

Branch: `ufunc_reduction_performance` vs `main`
Python: 3.14.3 release (GIL, pymalloc, Clang 21.1.4)
Platform: Linux x86_64
Date: 2026-03-27
Tool: pyperf (--fast mode)
Tests: 5555 passed, 60 skipped, 7 xfailed

## Commits

```
cdba07377d ENH: Skip __array_function__ dispatcher for exact ndarray args
b20b8d4a3e ENH: Additional reduction optimizations
             - Fast path in PyArray_FromAny for existing ndarray inputs
             - Cache legacy loop function on PyArrayMethodObject
             - Defer _get_bufsize_errmask to after fast reduction path
2d8397dfdc ENH: Fast path for full contiguous reductions
             - Bypass NpyIter and ReduceWrapper for axis=None, contiguous
```

## Results: main vs branch

| Benchmark | main | branch | Speedup |
|---|---|---|---|
| `np.sum(array_4)` | 1.44 us | 929 ns | **1.55x** |
| `np.sum(array_100)` | 1.45 us | 923 ns | **1.57x** |
| `np.prod(array_100)` | 1.47 us | 927 ns | **1.59x** |
| `np.sum(int_100)` | 1.47 us | 922 ns | **1.59x** |
| `np.sum(10x10)` | 1.49 us | 930 ns | **1.60x** |
| `np.sum(2x2_float)` | 1.49 us | 910 ns | **1.63x** |
| `np.sum(2x2_int)` | 1.47 us | 915 ns | **1.61x** |
| `np.any(bool_100)` | 1.31 us | 750 ns | **1.75x** |
| `np.max(array_100)` | 1.47 us | 1.35 us | **1.08x** |
| `np.sum(10x10,axis=0)` | 1.66 us | 1.51 us | **1.10x** |
| `np.sum(10x10,axis=1)` | 1.59 us | 1.48 us | **1.08x** |
| `np.sin(array_100)` | 765 ns | 744 ns | **1.03x** |
| **Geometric mean** | | | **1.41x** |

## Analysis

### What the fast path bypasses (for full contiguous reductions)

- `NpyIter_AdvancedNew` (993 instr) — full iterator setup
- `npyiter_*` helpers (1067 instr) — buffer copies, axis data, temp arrays
- `raw_array_assign_scalar` (176 instr) — casting pipeline for identity
- `PyUFunc_ReduceWrapper` (243 instr) — wrapper overhead

### What PyArray_FromAny fast path bypasses (for all callers)

- `PyArray_DiscoverDTypeAndShape` (176 instr) — dtype/shape discovery
- `PyArray_CanCastArrayTo` (186 instr) — casting check
- `PyArray_FromArray` (77 instr) — array conversion
- Total: ~680 instructions saved per call

### What loop cache bypasses

- `PyUFunc_DefaultLegacyInnerLoopSelector` (242 instr) — linear search
- `get_wrapped_legacy_ufunc_loop` auxdata allocation

### Operations NOT improved by fast path

- `np.max/np.min` — no identity value, cannot use fast reduction
- `np.sum(axis=0/1)` — axis reductions need outer loop, use NpyIter
- Reductions on non-contiguous arrays
- Reductions with `out=`, `where=`, `initial=`, `keepdims=True`

### What dispatcher fast path bypasses (for all dispatched functions)

- `relevant_arg_func` Python call (~150 instr) — no longer called
- `get_implementing_args_and_methods` (~87 instr) — no longer called
- `is_default_array_function` checks (~69 instr) — no longer called
- Total: ~300 instructions saved per dispatched call

This benefits not just reductions but ALL dispatched numpy functions
(np.sort, np.reshape, np.concatenate, etc.) when inputs are exact ndarrays.

### Remaining overhead (for fast-path reductions, ~923ns)

| Category | Est. instr/call | Notes |
|---|---|---|
| `default_impl` Python call | ~200 | sum() → _wrapreduction() → ufunc.reduce() |
| GenericReduction arg parsing | ~510 | Keyword parsing overhead |
| Promotion + resolve | ~340 | Could be cached further |
| Result (0-D array + scalar) | ~250 | Array alloc + PyArray_Scalar |
| Computation | ~273 | Actual pairwise sum |

## Benchmark script

See `bench_reduction_performance.py`.
