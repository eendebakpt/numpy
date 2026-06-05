# Plan: Fast path for simple reductions

## Goal

Add a trivial reduction fast path that bypasses NpyIter for the common case,
similar to `try_trivial_single_output_loop` for unary ufuncs.

Target: reduce `np.sum(np.ones(100))` from ~1380ns to ~500ns.

## Current bottleneck breakdown (np.sum(array_100), 1380ns)

| Cost | Instr/call | What |
|---|---|---|
| NpyIter setup/exec/dealloc | 2060 | Full iterator for 1 contiguous array |
| Initial value setup | 1119 | `raw_array_assign_scalar` casting pipeline |
| Reduction framework | 753 | `GenericReduction` + `ReduceWrapper` |
| Override/dispatch | 471 | `__array_function__`, arg parsing |
| Promotion | 422 | `promote_and_get_ufuncimpl` |
| Actual computation | 233 | `DOUBLE_pairwise_sum` |

## Scope of the fast path

### Entry conditions (all must be true)

1. `operation == UFUNC_REDUCE` (not accumulate or reduceat)
2. `out == NULL` (no pre-allocated output)
3. `wheremask == NULL` (no where mask)
4. `initial == NULL` (use identity, not user-provided initial)
5. `keepdims == 0`
6. Input array dtype matches the method's expected dtype (no casting)
7. Input array is contiguous (C or F)
8. Method has `get_reduction_initial` (has identity value)

### Supported cases

**Full reduction (axis=None):**
- 1-D contiguous array: `np.sum(a)` where a is 1-D → scalar result
- N-D contiguous array: `np.sum(a)` where a is C-contiguous → scalar result
  (treat as flat 1-D array of size `a.size`)

**Axis reduction on contiguous arrays:**
- C-contiguous, axis=-1 (last axis): inner loop over last axis, outer loop
  over remaining. This is the most common case for 2-D arrays.
- C-contiguous, axis=0: need to loop with stride. Still avoidable if we
  handle the transpose mentally.

For v1, focus on **full reduction (axis=None)** which is the simplest and
most common case. Axis reductions can be added later.

### What the fast path does

```c
// In PyUFunc_Reduce, before the current code:
if (fast_reduce_conditions_met) {
    // 1. Get initial value from cache (memcpy, ~10 instr)
    char initial_buf[sizeof(npy_clongdouble)];
    context.method->get_reduction_initial(&context, 0, initial_buf);

    // 2. Get strided loop
    npy_intp strides[3] = {elsize, 0, 0};  // in_stride, out_stride=0, out_stride=0
    method->get_strided_loop(&context, 1, 0, strides, &loop, &auxdata, &flags);

    // 3. Call loop directly: accumulate input into initial_buf
    //    The reduce loop signature is: loop(context, data, &count, strides, auxdata)
    //    where data = {result_ptr, input_ptr, result_ptr}
    npy_intp count = PyArray_SIZE(arr);
    char *data[3] = {initial_buf, PyArray_BYTES(arr), initial_buf};
    loop(&context, data, &count, strides, auxdata);

    // 4. Create scalar result from initial_buf
    return PyArray_Scalar(initial_buf, descrs[0], NULL);
}
```

Wait — the reduce loop has a different calling convention. Let me check.

### Reduce loop calling convention

The reduce loop is called with:
- `data[0]` = output (accumulator, read-write)
- `data[1]` = input
- `data[2]` = output (same as data[0] for in-place reduction)
- `strides[0]` = output stride (0 for scalar accumulator)
- `strides[1]` = input stride
- `strides[2]` = output stride (0)

For contiguous float64 input:
- `strides = {0, 8, 0}`
- `data = {&accumulator, input_data, &accumulator}`
- `count = array_size`

The accumulator is initialized with the identity value, then the loop
processes all elements.

## Common reduction operations affected

| Function | Ufunc | Identity | Notes |
|---|---|---|---|
| `np.sum` | `np.add` | 0 | Most common |
| `np.prod` | `np.multiply` | 1 | |
| `np.max` | `np.maximum` | -inf | No identity for int types |
| `np.min` | `np.minimum` | +inf | No identity for int types |
| `np.any` | `np.logical_or` | False | |
| `np.all` | `np.logical_and` | True | |
| `np.add.reduce` | `np.add` | 0 | Direct ufunc call |

All of these would benefit from the fast path.

## Implementation location

Add the fast path inside `PyUFunc_Reduce()` in `ufunc_object.c`,
after the `reducelike_promote_and_resolve` call (which gives us the
method and descriptors) but before `PyUFunc_ReduceWrapper`.

## FP error handling

The fast path still needs to:
1. Clear FP status before the loop (unless `NPY_METH_NO_FLOATINGPOINT_ERRORS`)
2. Check FP status after the loop
3. Call `_check_ufunc_fperr` if errors occurred

## Estimated savings

| Component | Current | Fast path | Savings |
|---|---|---|---|
| NpyIter_AdvancedNew | 993 | 0 | 993 |
| npyiter_* helpers | 1067 | 0 | 1067 |
| raw_array_assign_scalar | 176 | memcpy (~5) | 171 |
| reduce_loop overhead | 79 | ~20 | 59 |
| Result creation | PyArray (123) | PyArray_Scalar (67) | 56 |
| **Total savings** | | | **~2346** |

Expected: ~1380ns → ~500ns for `np.sum(array_100)`.

## Benchmark script

See `bench_reduction_performance.py`.
