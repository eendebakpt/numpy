Faster ``import numpy``
-----------------------
Importing NumPy is about 20-25% faster. Modules that are only needed for rarely
used functionality (``inspect``, ``ctypes``, ``pickle``, ``platform``,
``textwrap``, ``numpy.linalg`` and most of ``numpy._typing``) are now imported
on first use, docstrings are attached using a C implementation of
``inspect.cleandoc``, and ``__cpu_targets_info__``/``__cpu_features__`` of
``numpy._core._multiarray_umath`` are built on first access.
