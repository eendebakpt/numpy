# Callgrind Analysis: `np.abs(x)` with `x = np.array([1., 2.])`

Branch: `ufunc_tuple_elimination_v2`
Date: 2026-03-15
Python: 3.15t-dev-debug (free-threaded debug build)
Iterations: 20,000

## Self cost (instruction count per call)

| Function | Ir/call | % of ufunc total |
|---|---|---|
| `ufunc_generic_fastcall` | 722 | 12.3% |
| `PyUFunc_DefaultLegacyInnerLoopSelector` | 234 | 4.0% |
| `PyArray_NewFromDescr_int` | 159 | 2.7% |
| `promote_and_get_ufuncimpl` | 147 | 2.5% |
| `resolve_descriptors` | 111 | 1.9% |
| `PyArrayIdentityHash_GetItem` | 92 | 1.6% |
| `DOUBLE_absolute` (actual work) | 75 | 1.3% |
| `PyUFunc_CheckOverride` | 71 | 1.2% |
| `npy_find_array_wrap` | 68 | 1.2% |
| `_clear_array_attributes` | 66 | 1.1% |
| `get_wrapped_legacy_ufunc_loop` | 64 | 1.1% |
| `simple_legacy_resolve_descriptors` | 57 | 1.0% |
| `npy_apply_wrap` | 53 | 0.9% |
| `_ufunc_setup_flags` | 48 | 0.8% |
| `npy_clear_floatstatus_barrier` | 46 | 0.8% |
| `fetch_curr_extobj_state` | 40 | 0.7% |
| `_get_bufsize_errmask` | 34 | 0.6% |
| `_check_ufunc_fperr` | 30 | 0.5% |

## Inclusive cost

`ufunc_generic_fastcall` inclusive: **5,873 Ir/call** (117.5M / 20K).
The actual computation (`DOUBLE_absolute`) is only 75 Ir — just **1.3%** of the total ufunc overhead.

## Key observations

1. **`ufunc_generic_fastcall` self cost (722 Ir)** is the largest single contributor.
   This is the argument parsing, workspace allocation, and cleanup logic.

2. **`PyUFunc_DefaultLegacyInnerLoopSelector` (234 Ir)** is surprisingly expensive —
   it's called to select the inner loop every time, even though the answer is always
   the same for a given dtype combination. This would be a good caching target.

3. **`promote_and_get_ufuncimpl` (147 Ir self) + `promote_and_get_info_and_ufuncimpl` (36 Ir)**
   — promotion/dispatch is ~183 Ir total. The `PyArrayIdentityHash_GetItem` (92 Ir) is the
   hash lookup inside it.

4. **`resolve_descriptors` (111 Ir) + `simple_legacy_resolve_descriptors` (57 Ir)**
   — descriptor resolution is ~168 Ir.

5. **`PyUFunc_CheckOverride` (71 Ir)** — checking for `__array_ufunc__` overrides on every
   call, even for plain ndarrays, costs non-trivially.

6. **Array allocation/deallocation** (`PyArray_NewFromDescr_int` 159 +
   `_clear_array_attributes` 66 + `npy_alloc_cache_dim` 34 + `npy_free_cache_dim` 34)
   — output array creation is ~293 Ir.

7. **FP error checking** (`npy_clear_floatstatus_barrier` 46 + `_check_ufunc_fperr` 30 +
   `fetch_curr_extobj_state` 40 + `_get_bufsize_errmask` 34) — 150 Ir for error state management.

## Deep dive: `PyUFunc_DefaultLegacyInnerLoopSelector`

### How it works

The function performs a **linear scan** over `ufunc->types` to find a matching type signature.
`ufunc->types` is a flat `char` array of size `ntypes * nargs`, where each group of `nargs`
entries represents one supported type signature (input type_nums followed by output type_nums).

### Data source

The `types` array is created in generated code (`__umath_generated.c`) from
`numpy/_core/code_generators/generate_umath.py`. It is assigned in
`PyUFunc_FromFuncAndDataAndSignatureAndIdentity` (ufunc_object.c) via `ufunc->types = types;`.

### Debug output for `np.abs(np.array([1., 2.]))`

```
=== PyUFunc_DefaultLegacyInnerLoopSelector ===
  ufunc: absolute, ntypes: 20, nargs: 2
  looking for: [numpy.float64(12), numpy.float64(12)]
  available signatures:
    [ 0]: numpy.bool(0), numpy.bool(0)
    [ 1]: numpy.int8(1), numpy.int8(1)
    [ 2]: numpy.uint8(2), numpy.uint8(2)
    [ 3]: numpy.int16(3), numpy.int16(3)
    [ 4]: numpy.uint16(4), numpy.uint16(4)
    [ 5]: numpy.int32(5), numpy.int32(5)
    [ 6]: numpy.uint32(6), numpy.uint32(6)
    [ 7]: numpy.int64(7), numpy.int64(7)
    [ 8]: numpy.uint64(8), numpy.uint64(8)
    [ 9]: numpy.longlong(9), numpy.longlong(9)
    [10]: numpy.ulonglong(10), numpy.ulonglong(10)
    [11]: numpy.float16(23), numpy.float16(23)
    [12]: numpy.float32(11), numpy.float32(11)
    [13]: numpy.float64(12), numpy.float64(12)       <-- match here
    [14]: numpy.longdouble(13), numpy.longdouble(13)
    [15]: numpy.timedelta64(22), numpy.timedelta64(22)
    [16]: numpy.complex64(14), numpy.float32(11)
    [17]: numpy.complex128(15), numpy.float64(12)
    [18]: numpy.clongdouble(16), numpy.longdouble(13)
    [19]: numpy.object_(17), numpy.object_(17)
  matched at index 13
```

### Analysis

- `absolute` has **20 type signatures** with nargs=2 (1 input + 1 output).
- For `float64` (type_num=12), the match is at **index 13** — the loop does
  **13 failed comparisons** before finding the match.
- The types are ordered by type_num (bool=0, int8=1, ..., float64=12), except
  float16 (type_num=23) is inserted at index 11 (between ulonglong and float32),
  breaking the sorted-by-type_num order.
- float64 is the most common dtype in practice but sits in the middle of the list.

### Potential optimizations

1. **Direct index lookup**: For "uniform" signatures (all args have the same type_num),
   use the type_num as a direct index into a lookup table instead of linear scan.
   Most built-in ufuncs have uniform signatures for the common types.

2. **Reorder types**: Put float64 and int64 first in the types array, since they are
   the most commonly used dtypes. This is a code generator change.

3. **Cache the result**: Since the same dtype combination always maps to the same loop,
   cache the last successful match on the ufunc object (or per type_num).

4. **Hash table**: The existing TODO in the code mentions this:
   `"TODO: There needs to be a loop selection acceleration structure, like a hash table."`
