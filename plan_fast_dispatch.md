# Plan: Eliminate Python overhead in reduction dispatch

## Background

### What `relevant_arg_func` does

Every `np.sum`, `np.any`, etc. is an `_ArrayFunctionDispatcher` object.
When called, `dispatcher_vectorcall` (C) does:

1. Calls `self->relevant_arg_func(args, kwargs)` — a **Python function** like
   `_sum_dispatcher(a, axis, dtype, out, ...)` that returns `(a, out)`.
   Purpose: identify which arguments might implement `__array_function__`.

2. Calls `get_implementing_args_and_methods` — checks each returned arg
   for `__array_function__` overrides.

3. If no overrides (the >99% case for plain ndarrays): calls
   `self->default_impl(args, kwargs)` — another **Python function** like
   `sum(a, axis, ...)` which calls `_wrapreduction(a, np.add, 'sum', ...)`.

4. `_wrapreduction` checks `type(obj) is not ndarray`, then calls
   `ufunc.reduce(obj, axis, dtype, out, **passkwargs)` — finally entering C.

### Cost breakdown per `np.sum(a)` call

| Step | Instr | What |
|---|---|---|
| `relevant_arg_func` (Python call) | ~150 | Call `_sum_dispatcher`, create tuple, iterate |
| `get_implementing_args_and_methods` | ~87 | Check `__array_function__` on each arg |
| `is_default_array_function` | ~69 | Verify no override |
| `default_impl` (Python call) | ~200 | Call `sum()` → `_wrapreduction()` → `ufunc.reduce()` |
| **Total Python dispatch** | **~500** | |

### What `_wrapreduction` does (Python)

```python
def _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs):
    passkwargs = {k: v for k, v in kwargs.items() if v is not np._NoValue}
    if type(obj) is not mu.ndarray:
        # try obj.sum(), obj.prod(), etc.
        ...
    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
```

For exact ndarrays, it just filters `_NoValue` kwargs and calls `ufunc.reduce`.

## Options considered

### Option 1: Convert np.sum/any/all to C

Rewrite the Python `sum()`, `any()`, `all()` functions as C functions
that call `PyUFunc_GenericReduction` directly.

- **Pro**: Eliminates all Python overhead (~500 instr)
- **Con**: High maintenance cost. Each function has docstrings, deprecation
  warnings, generator handling, and the `_wrapreduction` subclass protocol.
  Duplicating this in C is error-prone.

### Option 2: C fast path in `dispatcher_vectorcall`

Add a check at the top of `dispatcher_vectorcall`: if all positional args
are exact ndarrays (or scalars), skip the `relevant_arg_func` Python call
and go directly to `default_impl`.

- **Pro**: Benefits all dispatched functions, not just reductions
- **Con**: Still calls `default_impl` (Python). Saves ~150 instr, not ~500.
  Also, `relevant_arg_func` can extract non-positional args (like `out=`),
  so skipping it may miss override-implementing keyword arguments.

### Option 3: Implement `_wrapreduction` in C  ← CHOSEN

Replace the Python `_wrapreduction` and `_wrapreduction_any_all` with C
functions. The dispatcher's `default_impl` still gets called (Python `sum()`),
but `_wrapreduction` becomes a fast C call instead of Python.

Then, add a fast path in `dispatcher_vectorcall` for the common case:
when the dispatcher has a known C reduction pattern and all args are exact
ndarrays, bypass both `relevant_arg_func` AND `default_impl` and call
`ufunc.reduce` directly from C.

- **Pro**: Eliminates ~350 instr of Python overhead for the common case
- **Con**: Need to handle `_NoValue` filtering and subclass protocol in C

### Option 4: Make `default_impl` callable as C for known patterns

Store a C function pointer on the dispatcher for known reduction patterns.
When the fast path is taken, call the C function directly.

- **Pro**: Clean separation, flexible
- **Con**: New infrastructure needed

## Chosen approach: Option 3 (hybrid)

### Step 1: C `_wrapreduction` (~100 instr savings)

Move `_wrapreduction` to C. This eliminates:
- Python dict comprehension for `passkwargs` filtering
- Python `type(obj) is not ndarray` check
- Python `ufunc.reduce()` call overhead

### Step 2: Fast path in `dispatcher_vectorcall` (~250 instr savings)

For reduction dispatchers where:
- First positional arg is an exact ndarray
- No `out=` keyword (or `out=None`)

Skip `relevant_arg_func` entirely and call `default_impl` directly.
This is safe because for exact ndarrays, `get_implementing_args_and_methods`
will never find overrides.

### Estimated savings

| Step | Savings | Cumulative |
|---|---|---|
| Current (fast reduce + FromAny + loop cache) | - | 988 ns |
| Step 2: skip relevant_arg_func | ~150 instr (~25ns) | ~963 ns |
| Step 1: C _wrapreduction | ~200 instr (~35ns) | ~928 ns |
| **Total** | ~350 instr (~60ns) | **~928 ns** |

## Affected functions

All functions in `fromnumeric.py` that use `_wrapreduction`:
- `np.sum` → `_wrapreduction(a, np.add, 'sum', ...)`
- `np.prod` → `_wrapreduction(a, np.multiply, 'prod', ...)`
- `np.max/amax` → `_wrapreduction(a, np.maximum, 'max', ...)`
- `np.min/amin` → `_wrapreduction(a, np.minimum, 'min', ...)`
- `np.any` → `_wrapreduction_any_all(a, np.logical_or, 'any', ...)`
- `np.all` → `_wrapreduction_any_all(a, np.logical_and, 'all', ...)`
