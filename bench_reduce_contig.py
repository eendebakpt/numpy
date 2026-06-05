"""Benchmark script for the contiguous-reduction fast path PR.

Targets the cases the fast path covers (full-axis reductions on a
contiguous array with an identity-having op and no casting/where/out)
plus two real-world callers that benefit indirectly:
  - ``np.array_equal(a, b)``: ``(a == b).all()``
  - ``np.allclose(a, b)``: ``np.all(<comparison>)``
plus a baseline (``np.sin``) that does NOT take the fast path so we can
spot any layout-noise floor.
"""
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

# Boolean reductions — the underlying op is fast so dispatch overhead matters most
abool20 = np.ones(20, dtype=bool)
runner.bench_func('np.any(bool_20)', np.any, abool20)
runner.bench_func('np.all(bool_20)', np.all, abool20)

# Indirect beneficiaries — real numpy functions that internally do .all() on
# the result of an elementwise comparison (which produces a contiguous bool).
ai = np.arange(20, dtype=np.float64)
bi = ai.copy()
runner.bench_func('np.array_equal(a,b)', np.array_equal, ai, bi)
runner.bench_func('np.allclose(a,b)', np.allclose, ai, bi)

# Strided 1-D arrays (newly covered after switching to PyArray_TRIVIALLY_ITERABLE).
# These are non-contiguous slices of a 1-D buffer; previously they fell to the
# slow path because PyArray_ISCARRAY_RO/ISFARRAY_RO required NPY_ARRAY_C/F_CONTIGUOUS.
a200_strided = np.ones(200)[::2]   # length 100, stride 16 bytes
runner.bench_func('np.sum(strided_1d_100)', np.sum, a200_strided)
runner.bench_func('np.all(strided_bool_100)', np.all, np.ones(200, dtype=bool)[::2])

# Identity-less reductions (newly covered after seeding the accumulator with
# arr[0]).  np.max/np.min and the maximum/minimum.reduce family fall here.
runner.bench_func('np.max(array_20)', np.max, a20)
runner.bench_func('np.min(array_20)', np.min, a20)
runner.bench_func('np.maximum.reduce(array_100)', np.maximum.reduce, a100)
runner.bench_func('np.fmax.reduce(array_100)', np.fmax.reduce, a100)

# Cases that should NOT take the fast path (sanity / regression watch)
runner.bench_func('np.sum(10x10,axis=0)', np.sum, a10x10, 0)     # axis-reduction

# Pure baseline: ufunc, no reduction → not affected by this PR
runner.bench_func('np.sin(array_20)', np.sin, a20)
