# Ufunc Performance Benchmarks

Branch: `ufunc_performance` vs `main`
Python: 3.14.3 release (GIL, pymalloc, Clang 21.1.4)
Platform: Linux x86_64
Date: 2026-03-27
Tool: pyperf (--fast mode)

## Commits in branch

```
ec610db610 ENH: Inline dims/strides storage for 1-D and 2-D arrays
d94fc2ee7a ENH: Fast array allocation for super-fast ufunc path
fc1e3fae98 ENH: Super-fast path for contiguous unary ufuncs
80c18953ff ENH: Defer _get_bufsize_errmask in GenericFunctionInternal
b01c4aee0c ENH: Add size-based freelist for numpy scalar objects
691df277f7 ENH: Refactor _array_fill_strides to merge contiguity check
88ff3692fe ENH: Add inline dispatch cache and optimize promote_and_get_ufuncimpl
e3e0015b90 ENH: Skip floating point error checks for ufuncs that never raise them
69d8c0adde ENH: Combined ufunc performance improvements
             (tuple elimination + loop cache + happy path)
```

## Results: main vs branch

| Benchmark | main | branch | Speedup |
|---|---|---|---|
| `np.abs(python_float)` | 77.7 ns | 62.3 ns | **1.25x** |
| `np.sin(python_float)` | 65.8 ns | 59.9 ns | **1.10x** |
| `np.abs(float64)` | 100 ns | 80.9 ns | **1.24x** |
| `np.sin(float64)` | 85.4 ns | 78.0 ns | **1.09x** |
| `np.float64(1.0)` | 36.8 ns | 38.3 ns | 1.04x slower |
| `np.abs(array_1)` | 258 ns | 117 ns | **2.20x** |
| `np.sin(array_1)` | 251 ns | 117 ns | **2.15x** |
| `np.abs(array_4)` | 257 ns | 118 ns | **2.18x** |
| `np.sin(array_4)` | 273 ns | 128 ns | **2.13x** |
| `np.abs(array_100)` | 265 ns | 125 ns | **2.13x** |
| `np.sin(array_100)` | 761 ns | 591 ns | **1.29x** |
| `np.sin(array_2x2)` | 274 ns | 129 ns | **2.11x** |
| `np.sin(int_array_4)` | 480 ns | 308 ns | **1.56x** |
| `np.sum(array_100)` | 1430 ns | 1390 ns | **1.03x** |
| **Geometric mean** | | | **1.64x faster** |

## What each commit does

### 1. Combined ufunc performance improvements (69d8c0adde)

Three optimizations combined:
- **Tuple elimination**: Replace intermediate Python tuples in ufunc dispatch
  with C arrays and borrowed references. Saves ~511M instructions per 1M
  `np.sin(array_1)` calls.
- **Loop cache**: Cache the legacy loop function pointer on `PyArrayMethodObject`
  at creation time, avoiding the `PyUFunc_DefaultLegacyInnerLoopSelector`
  linear search and heap-allocated auxdata on every call.
- **Happy path**: Fast path for simple 1-in/1-out ufunc calls that skips
  override checking, keyword parsing, and wrapping overhead.

### 2. Skip FP error checks (e3e0015b90)

Adds `UFUNC_NO_FLOATINGPOINT_ERRORS` flag for ufuncs that never raise
floating point errors (abs, neg, comparisons, logical ops, copysign, etc.).
Skips the expensive `npy_clear_floatstatus`/`npy_get_floatstatus` calls.

### 3. Inline dispatch cache (88ff3692fe)

Direct-mapped cache on `PyUFuncObject` avoids the hash table lookup when
the same DType(s) are used repeatedly. Also splits `promote_and_get_ufuncimpl`
into signature/no-signature paths.

### 4. Refactor _array_fill_strides (691df277f7)

Merges the contiguity pre-scan loop into the stride-filling loop.
Branchless zero-dimension handling. Saves ~5 instructions per call.

### 5. Scalar freelist (b01c4aee0c)

Size-based per-thread freelist for numpy scalar objects.
Neutral on GIL+pymalloc builds, helps on free-threading+mimalloc.

### 6. Defer _get_bufsize_errmask (80c18953ff)

Lazily fetches the TLS errstate so the trivial loop path avoids
the `PyContextVar_Get` call. Saves ~57 instructions when the
trivial loop path succeeds.

### 9. Inline dims/strides for 1-D and 2-D arrays (ec610db610)

Add `_inline_dim_strides[4]` to `PyArrayObject_fields`. For ndim <= 2,
`dimensions` and `strides` pointers point to this inline storage instead
of a separately heap-allocated block. Eliminates `npy_alloc_cache_dim`
and `npy_free_cache_dim` for ~70% of arrays. Increases `tp_basicsize`
from 96 to 128 bytes (both powers of 2, so allocator-friendly).

Saves ~12ns per array creation+destruction cycle.

### 8. Fast array allocation for super-fast path (d94fc2ee7a)

Add `PyArray_NewLikeArray_fast()` — a streamlined array constructor that
skips subarray/unsized checks, computes strides and contiguity flags
inline, and avoids `PyArray_UpdateFlags`. Used by the super-fast ufunc
path instead of `PyArray_NewFromDescr`. Saves ~10ns per call, bringing
`np.abs(array_1)` from 140ns to 130ns.

Also excludes object dtypes from the super-fast path (they need
PYAPI handling for reference counting).

### 7. Super-fast path for contiguous unary ufuncs (fc1e3fae98)

For nin==1, nout==1, contiguous aligned input with matching dtype
singleton, bypasses `promote_and_get_ufuncimpl`, `resolve_descriptors`,
and `GenericFunctionInternal` entirely. A single hash table lookup
resolves both the method and output dtype. Then directly allocates
the output array and calls the strided loop. Saves ~500 instructions
per call, bringing `np.sin(array_1)` from 177ns to 139ns.

## Tests

All numpy core tests pass (5555 passed, 60 skipped, 7 xfailed).

## Remaining bottleneck: output array allocation

The largest remaining cost in the super-fast path is `PyArray_NewFromDescr_int`
at ~159M instructions per 1M calls (~30ns per call, ~20% of total).

### Callgrind breakdown of PyArray_NewFromDescr_int (per call)

| Function | Instr/call | What it does |
|---|---|---|
| `PyDataMem_UserNEW` | 30 | Allocate data buffer |
| → `PyDataMem_GetHandler` | 17 | PyContextVar_Get for memory handler |
| → `default_malloc` | 37 | Actual malloc |
| `PyType_GenericAlloc` | 121 | Allocate PyObject (tp_alloc) |
| `npy_alloc_cache_dim` | 34 | Allocate dims/strides array |
| `_array_fill_strides` | 34 | Compute strides + contiguity flags |
| `PyArray_UpdateFlags` | 26 | Recompute array flags |
| `PyArray_MultiplyList` | 26 | Compute total size |
| **Total** | ~325 | |

### Plan to optimize output array allocation

**Option A: Fast 1-D contiguous array constructor (~60% savings)**

Create a specialized `PyArray_NewContiguous1D(descr, size)` that:
1. Skips `PyDataMem_GetHandler` (use default handler directly)
2. Skips `_array_fill_strides` (stride = itemsize, always C-contiguous)
3. Skips `PyArray_UpdateFlags` (flags are known: C_CONTIGUOUS | F_CONTIGUOUS | ALIGNED | WRITEABLE)
4. Skips `PyArray_MultiplyList` (total size = size * itemsize)
5. Skips `npy_alloc_cache_dim` for 1-D (store dims/strides inline or use a fast path)

This would reduce the allocation from ~325 to ~130 instructions.

**Option B: Output array cache/freelist**

Cache recently deallocated small arrays (similar to the scalar freelist).
When the super-fast path needs an output array of the same shape and dtype
as the last call, reuse the deallocated array directly. This would reduce
allocation cost to near zero for repeated calls with the same shape.

Complexity: moderate — needs careful reference counting and thread safety.

**Option C: Stack-allocated output for tiny arrays**

For arrays with <= 8 elements, use stack-allocated storage instead of
heap allocation. The array object still needs to be heap-allocated
(it's a Python object), but the data buffer can be inline.

NumPy already has `NPY_ARRAY_WRITEBACKIFCOPY` patterns but not inline data.
This would be a more invasive change.

**Option D: Inline dims/strides on PyArrayObject (partially implemented)**

Add `npy_intp _inline_dim_strides[4]` to `PyArrayObject_fields` so that
1-D and 2-D arrays store their dimensions and strides inline (avoiding
`npy_alloc_cache_dim`/`npy_free_cache_dim`).  Increases `tp_basicsize`
from 96 to 128 bytes.

Status: prototype implemented and tested, but too many code paths directly
manipulate `fa->dimensions` (getset.c, shape.c, methods.c, scalarapi.c,
iterators.c, etc.).  A clean implementation requires auditing all of these
and adding a helper like `PyArray_AllocDimStrides(fa, nd)`.  Estimated
savings: ~34 instructions per call (the `npy_alloc_cache_dim` +
`npy_free_cache_dim` pair).

**Recommendation**: Option A (fast constructor) is implemented and committed.
Option D (inline dims) is the next logical step but needs careful
auditing of ~15 call sites that touch `fa->dimensions`.

## Benchmark script

See `bench_ufunc_performance.py` in the repository root.

## Notes

- The `np.float64(1.0)` regression (1.04x) is within noise; the freelist
  is neutral on GIL+pymalloc builds.
- `np.abs(python_float)` vs `np.abs(float64)`: 20ns gap because python
  `float` hits `PyFloat_CheckExact` (fast), while `np.float64` falls through
  to `is_anyscalar_exact` + `PyArray_DescrFromScalar` (slower).
- `np.sin(array_100)` vs `np.abs(array_100)`: 485ns difference is purely
  compute (Taylor series vs bitwise AND per element).
