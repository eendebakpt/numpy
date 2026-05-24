"""
Benchmarks for the ``PyArray_CheckFromAny`` fast path.

These exercise common callers of ``PyArray_CheckFromAny`` (and hence
``PyArray_CheckFromAny_int``) which pass an already-conforming ndarray.
With the fast path in place, no dtype/shape discovery should be needed.
"""
import numpy as np

from .common import Benchmark


class CheckFromAnyFastPath(Benchmark):
    """Small-array benchmarks where the FromAny overhead dominates."""

    def setup(self):
        # Small arrays so that per-call coercion overhead dominates.
        self.sorted_i64 = np.arange(16, dtype=np.int64)
        self.queries_i64 = np.arange(8, dtype=np.int64)
        self.bins_f64 = np.linspace(0, 1, 16, dtype=np.float64)
        self.samples_f64 = np.linspace(0, 1, 8, dtype=np.float64)
        self.counts_intp = np.arange(16, dtype=np.intp)
        self.part_i64 = np.arange(16, dtype=np.int64)[::-1].copy()
        self.a_f64 = np.arange(16, dtype=np.float64)
        self.b_f64 = np.arange(16, dtype=np.float64)
        self.mask = np.ones(16, dtype=bool)

    def time_searchsorted(self):
        np.searchsorted(self.sorted_i64, self.queries_i64)

    def time_digitize(self):
        np.digitize(self.samples_f64, self.bins_f64)

    def time_bincount(self):
        np.bincount(self.counts_intp)

    def time_partition(self):
        np.partition(self.part_i64, 3)

    def time_argpartition(self):
        np.argpartition(self.part_i64, 3)

    def time_interp(self):
        np.interp(self.samples_f64, self.bins_f64, self.bins_f64)

    def time_ufunc_where(self):
        np.add(self.a_f64, self.b_f64, out=self.a_f64, where=self.mask)


class CheckFromAnyFastPathLarge(Benchmark):
    """Same calls with larger arrays - amortizes coercion better."""

    def setup(self):
        self.sorted_i64 = np.arange(10_000, dtype=np.int64)
        self.queries_i64 = np.arange(1_000, dtype=np.int64)
        self.bins_f64 = np.linspace(0, 1, 1_000, dtype=np.float64)
        self.samples_f64 = np.linspace(0, 1, 1_000, dtype=np.float64)
        self.counts_intp = np.arange(10_000, dtype=np.intp)
        self.part_i64 = np.arange(10_000, dtype=np.int64)[::-1].copy()
        self.a_f64 = np.arange(10_000, dtype=np.float64)
        self.b_f64 = np.arange(10_000, dtype=np.float64)
        self.mask = np.ones(10_000, dtype=bool)

    def time_searchsorted_large(self):
        np.searchsorted(self.sorted_i64, self.queries_i64)

    def time_bincount_large(self):
        np.bincount(self.counts_intp)

    def time_partition_large(self):
        np.partition(self.part_i64, 1000)

    def time_ufunc_where_large(self):
        np.add(self.a_f64, self.b_f64, out=self.a_f64, where=self.mask)
