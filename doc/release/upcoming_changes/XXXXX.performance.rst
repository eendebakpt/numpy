Faster ``import numpy``
-----------------------
``import numpy`` no longer creates the ~1300 legacy inner-loop wrappers of the
built-in ufuncs up front; they are created on first use of a loop.  On Python
3.15+, ``numpy.linalg``, ``platform`` and the ``numpy.polynomial`` series
modules are imported lazily (PEP 810), on first use.
