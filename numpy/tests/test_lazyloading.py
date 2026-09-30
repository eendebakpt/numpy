import sys
import textwrap

import pytest

from numpy.testing import HAS_SUBPROCESSES
from numpy.testing._private.utils import run_subprocess


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
def test_lazy_load():
    # gh-22045. lazyload doesn't import submodule names into the namespace

    # Test within a new process, to ensure that we do not mess with the
    # global state during the test run (could lead to cryptic test failures).
    # This is generally unsafe, especially, since we also reload the C-modules.
    code = textwrap.dedent(r"""
        import sys
        from importlib.util import LazyLoader, find_spec, module_from_spec

        # create lazy load of numpy as np
        spec = find_spec("numpy")
        module = module_from_spec(spec)
        sys.modules["numpy"] = module
        loader = LazyLoader(spec.loader)
        loader.exec_module(module)
        np = module

        # test a subpackage import
        from numpy.lib import recfunctions  # noqa: F401

        # test triggering the import of the package
        np.ndarray
        """)
    run_subprocess((sys.executable, '-c', code))


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
def test_import_avoids_expensive_stdlib_modules():
    # These modules are only needed for rarely used functionality and are
    # imported on first use, to keep `import numpy` fast.
    code = textwrap.dedent(r"""
        import sys
        before = set(sys.modules)
        import numpy
        new = set(sys.modules) - before
        expensive = {"inspect", "ctypes", "pickle", "platform", "textwrap",
                     "numpy.linalg", "numpy._typing._char_codes"}
        print(sorted(new & expensive))
        """)
    p = run_subprocess([sys.executable, "-c", code])
    assert p.stdout.strip() == "[]"


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
@pytest.mark.skipif(sys.version_info < (3, 15),
                    reason="PEP 810 lazy imports need Python 3.15")
def test_global_lazy_imports():
    # Under `-X lazy_imports=all` (PEP 810) modules that are imported only for
    # their side effects (docstrings, distributor init) must still be executed.
    code = textwrap.dedent(r"""
        import sys
        import numpy as np
        assert np.arange(3).sum() == 3
        assert "numpy._distributor_init" in sys.modules
        assert np.ndarray.__doc__ and np.ndarray.sum.__doc__  # from _add_newdocs
        assert np.float64.__doc__  # from _add_newdocs_scalars
        assert np.linalg.norm([3, 4]) == 5
        print("ok")
        """)
    p = run_subprocess([sys.executable, "-X", "lazy_imports=all", "-c", code])
    assert p.stdout.strip() == "ok"
