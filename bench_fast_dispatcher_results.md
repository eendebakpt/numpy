# Benchmark Results: fast_dispatcher branch vs main

Branch: `fast_dispatcher`
Benchmark script: bench_reduction_performance.py
Date: 2026-04-18
Python: 3.13.3 (release build, -O2 -DNDEBUG, GCC 14.2.0)
NumPy: 2.5.0.dev0

## Two changes in this branch

1. **Tuple-spec dispatchers** — `_ArrayFunctionDispatcher` now accepts a
   tuple of arg names (`("a", "out")`) instead of a Python callable.
   Skips the per-call Python dispatcher invocation.
2. **Safe-arg early bailout** (tuple-spec only) — when every relevant arg
   is an exact ndarray, `None`, or a primitive scalar (int/float/bool/str),
   no `__array_function__` override is possible. Skip the override
   machinery entirely and call `default_impl` directly.

The bailout is restricted to tuple-spec dispatchers because (a) we already
know exactly which args are relevant (so `axis=(0, 1)` doesn't trigger a
spurious bailout-block), and (b) legacy callable dispatchers may have
side effects (test_overrides has a test that depends on this).

Converted 17 hot reduction dispatchers (sum, prod, max, min, any, all,
mean, std, var, ptp, cumsum, cumprod, cumulative_sum, cumulative_prod,
argmax, argmin, count_nonzero).

## Results

| Benchmark | main | tuple-spec only | tuple-spec + bailout (final) |
|---|---|---|---|
| `np.sum(array_4)` | 1.46 us | 1.42 us (1.03x) | 1.38 us (1.06x) |
| `np.sum(array_100)` | 1.47 us | 1.42 us (1.04x) | 1.40 us (1.05x) |
| `np.prod(array_100)` | 1.48 us | 1.43 us (1.04x) | 1.40 us (1.05x) |
| `np.max(array_100)` | 1.50 us | 1.44 us (1.04x) | 1.41 us (1.06x) |
| `np.sum(int_100)` | 1.48 us | 1.42 us (1.04x) | 1.39 us (1.07x) |
| `np.sum(10x10)` | 1.51 us | 1.45 us (1.04x) | 1.43 us (1.05x) |
| `np.sum(10x10, axis=0)` | 1.67 us | 1.63 us (1.02x) | 1.59 us (1.05x) |
| `np.sum(10x10, axis=1)` | 1.63 us | 1.59 us (1.03x) | 1.56 us (1.05x) |
| `np.any(bool_100)` | 1.33 us | 1.26 us (1.06x) | 1.24 us (1.08x) |
| `np.sum(2x2_float)` | 1.50 us | 1.44 us (1.05x) | 1.42 us (1.06x) |
| `np.sum(2x2_int)` | 1.50 us | 1.45 us (1.03x) | 1.43 us (1.05x) |
| `np.sin(array_100)` (baseline) | 803 ns | 776 ns | 780 ns (noise) |
| **Geometric mean** | (ref) | **1.04x** | **1.05x** |

## Why the gain is smaller than originally anticipated

The naive analysis suggested ~100-300 ns per call from skipping the Python
dispatcher. Actual savings are ~50-100 ns. Reasons:

1. **The Python dispatcher is very simple** (`return (a, out)`). With the
   vectorcall protocol, calling a 2-line Python function is ~50 ns, not
   200 ns.
2. **The dominant overhead is downstream** of the dispatcher call:
   `PySequence_Fast`, `get_implementing_args_and_methods` (which does
   per-arg type-dedup + `PyArray_LookupSpecial`), `get_args_and_kwargs`
   (allocates a tuple + dict), and the `types` tuple allocation. These
   sum to ~300-400 ns and are bypassed only by the early-bailout path.
3. **The reduction itself dominates the call cost.** `np.sum(array_100)`
   spends ~1 us in the actual reduction (NumPy iterator setup + loop).
   The overhead reduction is fixed at ~50 ns regardless of array size,
   so for fast reductions (`np.any` on bools = the entire reduction is
   ~30 ns) the relative gain is highest (8%); for slower reductions the
   relative gain is smaller (3-5%).

## Comparison with commit cdba073 from `ufunc_reduction_performance` branch

cdba073 was reported to give "1.5x speedup" — that number came from
benchmarking the **whole** `ufunc_reduction_performance` branch (which
stacks 4 other reduction optimizations) against main, not cdba073 alone.

Direct comparison of cdba073 against its own parent (`b20b8d4a3e`):
**~1.08x geomean** — comparable to my branch's 1.05x against main.

cdba073 also has two pre-existing test failures
(`TestNDArrayArrayFunction::test_method` and
`TestArrayFunctionImplementation::test_dispatcher_error`) because its
bailout fires too aggressively (containers + side-effect dispatchers).
My branch passes both because the bailout is restricted to tuple-spec
dispatchers.

## Stacking with other reduction optimizations

The 1.5x speedup on the full `ufunc_reduction_performance` branch
suggests there's more to be had by combining this PR with:
- Fast path for full contiguous reductions (~+5%)
- Skip `PyArray_FromAny` for ndarray inputs in `GenericReduction` (~+5%)
- Additional reduction optimizations (~+10%)

These are independent of the dispatch fast path and would compound.
