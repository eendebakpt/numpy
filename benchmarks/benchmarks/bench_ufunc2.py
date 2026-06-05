import numpy as np

from .common import Benchmark


class ScalarAstype(Benchmark):
    def setup(self):
        self.start = np.float64(5.0)
        self.dtype = np.float64(15.0).dtype
        self.dtype_other = np.float32(15.0).dtype

    def time_astype(self):
        self.start.astype(self.dtype)

    def time_astype_other(self):
        self.start.astype(self.dtype_other)

    def time_astype_v2(self):
        self.start.astype(np.float64)
