# PEP 810: deferred on Python 3.15+, ignored on older Python.
__lazy_modules__ = ["numpy.lib._npyio_impl"]

from ._npyio_impl import DataSource, NpzFile, __doc__  # noqa: F401
