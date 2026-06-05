# ENH: fast path for full contiguous reductions

### Summary

Adds a fast path in `PyUFunc_Reduce` for the common case of a full
reduction (`axis=None`) over a contiguous, aligned, non-object input
where the matching dtype equals the input dtype and the operation has
an identity value (e.g. `np.sum`, `np.prod`, `np.any`, `np.all`).


### Benchmark results

| Benchmark              | main    | branch              |
|------------------------|---------|---------------------|
| `np.sum(array_4)`      | 1.47 us | 1.09 us (1.36x)     |
| `np.sum(array_20)`     | 1.47 us | 1.10 us (1.34x)     |
| `np.sum(array_100)`    | 1.47 us | 1.10 us (1.34x)     |
| `np.prod(array_20)`    | 1.43 us | 1.07 us (1.34x)     |
| `np.sum(10x10)`        | 1.51 us | 1.11 us (1.36x)     |
| `np.sum(2x2)`          | 1.51 us | 1.09 us (1.38x)     |
| `np.any(bool_20)`      | 1.33 us | 926 ns (1.44x)      |
| `np.all(bool_20)`      | 1.33 us | 937 ns (1.42x)      |
| `np.array_equal(a,b)`  | 1.39 us | 1.06 us (1.31x)     |
| `np.allclose(a,b)`     | 12.4 us | 11.2 us (1.10x)     |
| `np.max(array_20)`     | n/s     | n/s (correctly bypassed: no identity) |
| `np.sum(10x10,axis=0)` | n/s     | n/s (correctly bypassed: axis-reduction) |
| `np.sin(array_20)`     | 400 ns  | 396 ns (≈noise, control) |
| **Geometric mean**     | (ref)   | **1.25x faster**    |

`np.array_equal` and `np.allclose` benefit indirectly: both internally
call `.all()` on a contiguous boolean.


<details><summary>Benchmark script</summary>

```python
"""Benchmark script for the contiguous-reduction fast path PR."""
import pyperf
import numpy as np

runner = pyperf.Runner()

# Direct fast-path targets (axis=None reductions, contiguous, identity op)
a4 = np.ones(4)
a20 = np.ones(20)
a100 = np.ones(100)
runner.bench_func('np.sum(array_4)', np.sum, a4)
runner.bench_func('np.sum(array_20)', np.sum, a20)
runner.bench_func('np.sum(array_100)', np.sum, a100)
runner.bench_func('np.prod(array_20)', np.prod, a20)

# 2D, full reduction (still contiguous → fast path)
a10x10 = np.ones((10, 10))
runner.bench_func('np.sum(10x10)', np.sum, a10x10)
runner.bench_func('np.sum(2x2)', np.sum, np.ones((2, 2)))

# Boolean reductions
abool20 = np.ones(20, dtype=bool)
runner.bench_func('np.any(bool_20)', np.any, abool20)
runner.bench_func('np.all(bool_20)', np.all, abool20)

# Indirect beneficiaries (numpy functions that internally do .all())
ai = np.arange(20, dtype=np.float64)
bi = ai.copy()
runner.bench_func('np.array_equal(a,b)', np.array_equal, ai, bi)
runner.bench_func('np.allclose(a,b)', np.allclose, ai, bi)

# Cases that should NOT take the fast path (sanity / regression watch)
runner.bench_func('np.max(array_20)', np.max, a20)
runner.bench_func('np.sum(10x10,axis=0)', np.sum, a10x10, 0)

# Pure baseline (ufunc, not a reduction → not affected by this PR)
runner.bench_func('np.sin(array_20)', np.sin, a20)
```

</details>
