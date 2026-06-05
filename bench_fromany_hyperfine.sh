#!/bin/bash
# Benchmark PyArray_FromAny fast path with hyperfine.
# Usage: ./bench_fromany_hyperfine.sh /path/to/python
#
# Each case is a tight loop of 500k iterations to amortize startup.
PYTHON="${1:-python3}"
N=500000

hyperfine \
  --warmup 2 \
  --min-runs 5 \
  --export-markdown /tmp/bench_fromany_hyperfine.md \
  -n "np.sum(array_100)" \
    "$PYTHON -c 'import numpy as np; a=np.ones(100)
for _ in range($N): np.sum(a)'" \
  -n "np.prod(array_100)" \
    "$PYTHON -c 'import numpy as np; a=np.ones(100)
for _ in range($N): np.prod(a)'" \
  -n "np.max(array_100)" \
    "$PYTHON -c 'import numpy as np; a=np.ones(100)
for _ in range($N): np.max(a)'" \
  -n "np.any(bool_100)" \
    "$PYTHON -c 'import numpy as np; a=np.ones(100,dtype=bool)
for _ in range($N): np.any(a)'" \
  -n "np.copyto(dst,src)" \
    "$PYTHON -c 'import numpy as np; a=np.ones(100); d=np.empty(100)
for _ in range($N): np.copyto(d,a)'" \
  -n "np.sin(a,out=out)" \
    "$PYTHON -c 'import numpy as np; a=np.ones(100); o=np.empty(100)
for _ in range($N): np.sin(a,out=o)'" \
  -n "np.sin(array_100) [no FromAny]" \
    "$PYTHON -c 'import numpy as np; a=np.ones(100)
for _ in range($N): np.sin(a)'" \
  -n "np.count_nonzero [misses fast path]" \
    "$PYTHON -c 'import numpy as np; a=np.ones(100)
for _ in range($N): np.count_nonzero(a)'" \
  -n "np.where(cond,a,b) [misses fast path]" \
    "$PYTHON -c 'import numpy as np; a=np.ones(100); b=np.zeros(100); c=np.ones(100,dtype=bool)
for _ in range($N): np.where(c,a,b)'"
