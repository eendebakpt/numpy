# Fast Dispatcher Migration Inventory

Overview of all `_*_dispatcher` functions, grouped by migration category.

## Categories

- **Converted** (17): hot reduction dispatchers migrated in this branch.
- **Convertible** (188 remaining): return `(name1, name2, ...)` of named args; can
  be replaced with `@array_function_dispatch(("name1", "name2", ...))`.
- **Varargs** (14): use `*args` / `*varargs` pattern; need either a special
  fast-path variant or stay as Python dispatchers.
- **Complex** (19): have conditional logic in the body; must stay as Python
  dispatchers.

## Already converted (this branch)

| File | Name | Spec |
|---|---|---|
| fromnumeric.py | _sum_dispatcher | `("a", "out")` |
| fromnumeric.py | _argmax_dispatcher | `("a", "out")` |
| fromnumeric.py | _argmin_dispatcher | `("a", "out")` |
| fromnumeric.py | _any_dispatcher | `("a", "where", "out")` |
| fromnumeric.py | _all_dispatcher | `("a", "where", "out")` |
| fromnumeric.py | _cumulative_prod_dispatcher | `("x", "out")` |
| fromnumeric.py | _cumulative_sum_dispatcher | `("x", "out")` |
| fromnumeric.py | _cumsum_dispatcher | `("a", "out")` |
| fromnumeric.py | _ptp_dispatcher | `("a", "out")` |
| fromnumeric.py | _max_dispatcher | `("a", "out")` |
| fromnumeric.py | _min_dispatcher | `("a", "out")` |
| fromnumeric.py | _prod_dispatcher | `("a", "out")` |
| fromnumeric.py | _cumprod_dispatcher | `("a", "out")` |
| fromnumeric.py | _mean_dispatcher | `("a", "where", "out")` |
| fromnumeric.py | _std_dispatcher | `("a", "where", "out", "mean")` |
| fromnumeric.py | _var_dispatcher | `("a", "where", "out", "mean")` |
| numeric.py | _count_nonzero_dispatcher | `("a",)` |

## Convertible (remaining)

205 dispatchers listed in full detail; here's the breakdown by module:

### `numpy/_core/`
- **arrayprint.py** (3): `_array2string_dispatcher`, `_array_repr_dispatcher`, `_array_str_dispatcher`
- **defchararray.py** (1): `_binary_op_dispatcher`
- **einsumfunc.py** (1): `_einsum_path_dispatcher` — see varargs note below
- **fromnumeric.py** (25): non-reduction dispatchers — `_take`, `_reshape`, `_repeat`, `_put`, `_swapaxes`, `_transpose`, `_matrix_transpose`, `_partition`, `_argpartition`, `_sort`, `_argsort`, `_searchsorted`, `_resize`, `_squeeze`, `_diagonal`, `_trace`, `_ravel`, `_nonzero`, `_shape`, `_compress`, `_clip`, `_ndim`, `_size`, `_round`
- **function_base.py** (3): `_linspace`, `_logspace`, `_geomspace`
- **numeric.py** (19): `_zeros_like`, `_ones_like`, `_full`, `_full_like`, `_argwhere`, `_flatnonzero`, `_correlate`, `_convolve`, `_outer`, `_tensordot`, `_roll`, `_rollaxis`, `_moveaxis`, `_cross`, `_allclose`, `_isclose`, `_array_equal`, `_array_equiv`, `_astype`
- **shape_base.py** (5): 3 varargs (`_atleast_1d/2d/3d`), `_stack`, `_unstack`
- **strings.py** (13): string operations

### `numpy/lib/`
- **_arraypad_impl.py** (1): `_pad_dispatcher`
- **_arraysetops_impl.py** (11): unique, intersect/union/setdiff, isin
- **_function_base_impl.py** (~26): flip, rot90, average, copy, diff, interp, angle, unwrap, extract, place, cov, corrcoef, i0, sinc, median, percentile, quantile, trapezoid, delete, insert, append, digitize, + varargs (`_piecewise`, `_gradient`, `_meshgrid`)
- **_histograms_impl.py** (2): `_histogram_bin_edges`, `_histogram`
- **_index_tricks_impl.py** (2): `_ix_` (varargs), `_fill_diagonal`
- **_nanfunctions_impl.py** (14): all `_nan*` dispatchers
- **_npyio_impl.py** (2): `_save`, `_savetxt`
- **_polynomial_impl.py** (8): polynomial ops
- **_scimath_impl.py** (3): `_unary`, `_logn`, `_power`
- **_shape_base_impl.py** (10): take_along_axis, put_along_axis, apply_over_axes, expand_dims, array_split, split, hvdsplit, kron, tile + 1 vararg (`_apply_along_axis`)
- **_stride_tricks_impl.py** (3): `_sliding_window_view`, `_broadcast_to`, `_broadcast_arrays` (vararg)
- **_twodim_base_impl.py** (5): `_flip`, `_diag`, `_trilu`, `_vander`, `_trilu_indices_form`
- **_type_check_impl.py** (6): real, imag, is_type, nan_to_num, real_if_close + 1 vararg (`_common_type`)
- **_ufunclike_impl.py** (1): `_dispatcher`
- **recfunctions.py** (15): record array operations

### `numpy/linalg/`
- **_linalg.py** (25): `_tensorsolve`, `_solve`, `_tensorinv`, `_unary`, `_matrix_power`, `_cholesky`, `_outer`, `_qr`, `_eigvalsh`, `_svd`, `_svdvals`, `_cond`, `_matrix_rank`, `_pinv`, `_lstsq`, `_norm`, `_diagonal`, `_trace`, `_cross`, `_matmul`, `_tensordot`, `_matrix_transpose`, `_matrix_norm`, `_vector_norm`, `_vecdot`

### `numpy/fft/`
- **_helper.py** (1): `_fftshift`
- **_pocketfft.py** (2): `_fft`, `_fftn`

### `numpy/polynomial/`
- **polynomial.py** (2): `_polyval2d`, `_polygrid2d`

## Varargs (need special handling)

These use `*args`-style collection. The current fast path only supports named
positional args. To migrate, either (a) extend the C spec to denote "collect
all positional args from position N onwards", or (b) keep as Python dispatchers.

| File | Name | Pattern |
|---|---|---|
| einsumfunc.py | _einsum_path_dispatcher | `*operands` |
| einsumfunc.py | _einsum_dispatcher | `*operands` (actually complex — see below) |
| shape_base.py | _atleast_1d/2d/3d_dispatcher | `*arys` |
| _function_base_impl.py | _piecewise_dispatcher | `*args` (complex) |
| _function_base_impl.py | _gradient_dispatcher | `*varargs` (complex) |
| _function_base_impl.py | _meshgrid_dispatcher | `*xi` |
| _index_tricks_impl.py | _ix__dispatcher | `*args` |
| _npyio_impl.py | _savez_dispatcher | `*args` (complex) |
| _npyio_impl.py | _savez_compressed_dispatcher | `*args` (complex) |
| _shape_base_impl.py | _apply_along_axis_dispatcher | `*args` |
| _stride_tricks_impl.py | _broadcast_arrays_dispatcher | `*args` |
| _type_check_impl.py | _common_type_dispatcher | `*arrays` |

## Complex (must stay as Python dispatchers)

Conditional logic in body (e.g., include arg only if non-None, iterate nested
structures). 19 total:

| File | Name | Reason |
|---|---|---|
| einsumfunc.py | _einsum_dispatcher | complex args structure |
| fromnumeric.py | _choose_dispatcher | iterates choices |
| shape_base.py | _arrays_for_stack_dispatcher | iterates arrays |
| shape_base.py | _vhstack_dispatcher | iterates |
| shape_base.py | _block_dispatcher | recursive |
| _function_base_impl.py | _piecewise_dispatcher | conditional |
| _function_base_impl.py | _select_dispatcher | iterates condlist/choicelist |
| _function_base_impl.py | _gradient_dispatcher | conditional |
| _histograms_impl.py | _histogramdd_dispatcher | conditional |
| _npyio_impl.py | _savez_dispatcher | iterates |
| _npyio_impl.py | _savez_compressed_dispatcher | iterates |
| _shape_base_impl.py | _column_stack_dispatcher | iterates |
| _shape_base_impl.py | _dstack_dispatcher | iterates |
| _twodim_base_impl.py | _histogram2d_dispatcher | conditional |
| recfunctions.py | _append_fields_dispatcher | conditional |
| recfunctions.py | _rec_append_fields_dispatcher | conditional |
| linalg/_linalg.py | _multidot_dispatcher | iterates |
| polynomial/polynomial.py | _polyvalnd_dispatcher | iterates |
| overrides.py | array_function_from_dispatcher | not actually a dispatcher |

## Migration plan for follow-up PRs

1. **This PR**: convert hot reductions (~17 done), establish the API.
2. **PR 2**: convert all remaining "simple tuple" dispatchers in
   `fromnumeric.py`, `numeric.py`, `function_base.py` (~50).
3. **PR 3**: convert `_nan*` and other lib dispatchers (~60).
4. **PR 4**: convert `linalg`, `fft`, `strings`, `polynomial`, `recfunctions`
   (~70).
5. **PR 5** (optional): extend C fast path to support varargs — "collect all
   positional args from position N, minus a known keyword-only tail".
6. **Deprecation**: once ~90% are converted, consider deprecating the
   callable-dispatcher form of `array_function_dispatch`. Any external users
   would need to migrate to tuple specs.

## Deprecation considerations

`numpy._core.overrides.array_function_dispatch` is technically private
(`_core` namespace), but has been stable for years and may be used by:
- `array_api_compat`
- `dask`, `cupy`, `jax`, `pytorch`'s ndarray shims
- `numpy.ma` (internally)

If deprecating the callable form, a one-release cycle of warning with a
migration guide is probably sufficient given the private namespace.
