#ifndef _NPY_DISPATCHING_H
#define _NPY_DISPATCHING_H

#define _UMATHMODULE

#include <numpy/ufuncobject.h>
#include "array_method.h"

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT int
PyUFunc_AddLoop(PyUFuncObject *ufunc, PyObject *info, int ignore_duplicate);

NPY_NO_EXPORT int
PyUFunc_AddLoopFromSpec_int(PyObject *ufunc, PyArrayMethod_Spec *spec, int priv);

NPY_NO_EXPORT int
PyUFunc_AddLoopsFromSpecs(PyUFunc_LoopSlot *slots);

NPY_NO_EXPORT PyArrayMethodObject *
promote_and_get_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[],
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[],
        npy_bool force_legacy_promotion,
        npy_bool promote_pyscalars,
        npy_bool ensure_reduce_compatible);

/*
 * True for a not yet materialized legacy loop: `(DType_tuple, None)` or
 * `(DType_tuple, capsule)` with the capsule holding its indexed loop.
 */
#define NPY_LEGACY_INDEXED_LOOP_CAPSULE "numpy._legacy_indexed_loop"

static inline int
npy_loop_info_is_placeholder(PyObject *info)
{
    PyObject *item = PyTuple_GET_ITEM(info, 1);
    return item == Py_None
            || PyCapsule_IsValid(item, NPY_LEGACY_INDEXED_LOOP_CAPSULE);
}

NPY_NO_EXPORT int
set_legacy_indexed_loop(PyUFuncObject *ufunc, int typenum, int nargs,
        PyArrayMethod_StridedLoop *indexed_loop);

NPY_NO_EXPORT int
add_legacy_loop_placeholder(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *operation_dtypes[]);

NPY_NO_EXPORT PyObject *
npy_materialize_legacy_loop(PyUFuncObject *ufunc, PyObject *info);

NPY_NO_EXPORT PyObject *
add_and_return_legacy_wrapping_ufunc_loop(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *operation_dtypes[], int ignore_duplicate);

NPY_NO_EXPORT int
default_ufunc_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[]);

NPY_NO_EXPORT int
object_only_ufunc_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *NPY_UNUSED(op_dtypes[]),
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[]);

NPY_NO_EXPORT int
install_logical_ufunc_promoter(PyObject *ufunc);


#ifdef __cplusplus
}
#endif

#endif  /*_NPY_DISPATCHING_H */
