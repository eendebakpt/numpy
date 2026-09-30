Faster ``import numpy`` on Python 3.15+
---------------------------------------
On Python 3.15+, ``numpy.linalg``, ``platform`` and the ``numpy.polynomial``
series modules are imported lazily (PEP 810), on first use.
