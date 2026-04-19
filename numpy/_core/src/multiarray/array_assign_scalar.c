/*
 * This file implements assignment from a scalar to an ndarray.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/ndarraytypes.h>
#include "numpy/npy_math.h"

#include "npy_config.h"


#include "convert_datatype.h"
#include "methods.h"
#include "shape.h"
#include "lowlevel_strided_loops.h"

#include "array_assign.h"
#include "dtype_transfer.h"
#include "alloc.h"

#include "umathmodule.h"

/*
 * Assigns the scalar value to every element of the destination raw array.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_scalar(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, NPY_CASTING casting)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned;

    NPY_BEGIN_THREADS_DEF;

    /* Check both uint and true alignment */
    aligned = raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   npy_uint_alignment(dst_dtype->elsize)) &&
              raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   dst_dtype->alignment) &&
              npy_is_aligned(src_data, npy_uint_alignment(src_dtype->elsize) &&
              npy_is_aligned(src_data, src_dtype->alignment));

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareOneRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it) < 0) {
        return -1;
    }

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetDTypeTransferFunction(aligned,
                        0, dst_strides_it[0],
                        src_dtype, dst_dtype,
                        0,
                        &cast_info, &flags) != NPY_SUCCEED) {
        return -1;
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier(src_data);
    }
    if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    if (((int)casting & NPY_SAME_VALUE_CASTING_FLAG) > 0) {
        cast_info.context.flags |= NPY_SAME_VALUE_CONTEXT_FLAG;
    }

    npy_intp strides[2] = {0, dst_strides_it[0]};

    int result = 0;
    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        char *args[2] = {src_data, dst_data};
        result = cast_info.func(&cast_info.context,
                args, &shape_it[0], strides, cast_info.auxdata);
        if (result < 0) {
            goto fail;
        }
    } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord,
                            shape_it, dst_data, dst_strides_it);

    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier(src_data);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }

    return 0;
fail:
    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);
    return -1;
}

/*
 * Assigns the scalar value to every element of the destination raw array
 * where the 'wheremask' value is True.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_scalar(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp const *wheremask_strides, NPY_CASTING casting)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp wheremask_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned;

    NPY_BEGIN_THREADS_DEF;

    /* Check both uint and true alignment */
    aligned = raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   npy_uint_alignment(dst_dtype->elsize)) &&
              raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   dst_dtype->alignment) &&
              npy_is_aligned(src_data, npy_uint_alignment(src_dtype->elsize) &&
              npy_is_aligned(src_data, src_dtype->alignment));

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    wheremask_data, wheremask_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &wheremask_data, wheremask_strides_it) < 0) {
        return -1;
    }

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                        0, dst_strides_it[0], wheremask_strides_it[0],
                        src_dtype, dst_dtype, wheremask_dtype,
                        0,
                        &cast_info, &flags) != NPY_SUCCEED) {
        return -1;
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier(src_data);
    }
    if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }
    if (((int)casting & NPY_SAME_VALUE_CASTING_FLAG) != 0) {
        cast_info.context.flags |= NPY_SAME_VALUE_CONTEXT_FLAG;
    }

    npy_intp strides[2] = {0, dst_strides_it[0]};
    int result = 0;

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        PyArray_MaskedStridedUnaryOp *stransfer;
        stransfer = (PyArray_MaskedStridedUnaryOp *)cast_info.func;

        char *args[2] = {src_data, dst_data};
        result = stransfer(&cast_info.context,
                args, &shape_it[0], strides,
                (npy_bool *)wheremask_data, wheremask_strides_it[0],
                cast_info.auxdata);
        if (result < 0) {
            goto fail;
        }
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            wheremask_data, wheremask_strides_it);

    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier(src_data);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }

    return 0;

fail:
    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);
    return -1;
}

/*
 * Assigns a scalar value specified by 'src_dtype' and 'src_data'
 * to elements of 'dst'.
 *
 * dst: The destination array.
 * src_dtype: The data type of the source scalar.
 * src_data: The memory element of the source scalar.
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the assignment violates this
 *          casting rule.
 *
 * This function is implemented in array_assign_scalar.c.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignRawScalar(PyArrayObject *dst,
                        PyArray_Descr *src_dtype, char *src_data,
                        PyArrayObject *wheremask,
                        NPY_CASTING casting)
{
    /* Scratch for aligned/cast src; NULL means "not allocated". */
    NPY_DEFINE_WORKSPACE(tmp_src_data, long double, 4);
    tmp_src_data = NULL;

    if (PyArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
        return -1;
    }

    /* Check the casting rule */
    if (!PyArray_CanCastTypeTo(src_dtype, PyArray_DESCR(dst), casting)) {
        npy_set_invalid_cast_error(
                src_dtype, PyArray_DESCR(dst), casting, NPY_TRUE);
        return -1;
    }

    /*
     * Make a copy of the src data if it's a different dtype than 'dst'
     * or isn't aligned, and the destination we're copying to has
     * more than one element. To avoid having to manage object lifetimes,
     * we also skip this if 'dst' has an object dtype.
     */
    if ((!PyArray_EquivTypes(PyArray_DESCR(dst), src_dtype) ||
            !(npy_is_aligned(src_data, npy_uint_alignment(src_dtype->elsize)) &&
              npy_is_aligned(src_data, src_dtype->alignment))) &&
                    PyArray_SIZE(dst) > 1 &&
                    !PyDataType_REFCHK(PyArray_DESCR(dst))) {
        size_t n_elems = (PyArray_ITEMSIZE(dst) + sizeof(long double) - 1)
                         / sizeof(long double);
        NPY_INIT_WORKSPACE_ZEROED(tmp_src_data, long double, 4, n_elems);
        if (tmp_src_data == NULL) {
            goto fail;
        }

        if (PyArray_CastRawArrays(1, src_data, (char *)tmp_src_data, 0, 0,
                            src_dtype, PyArray_DESCR(dst), 0) != NPY_SUCCEED) {
            goto fail;
        }

        /* Replace src_data/src_dtype */
        src_data = (char *)tmp_src_data;
        src_dtype = PyArray_DESCR(dst);
    }

    if (wheremask == NULL) {
        /* A straightforward value assignment */
        /* Do the assignment with raw array iteration */
        if (raw_array_assign_scalar(PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                src_dtype, src_data, casting) < 0) {
            goto fail;
        }
    }
    else {
        npy_intp wheremask_strides[NPY_MAXDIMS];

        /* Broadcast the wheremask to 'dst' for raw iteration */
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(wheremask), PyArray_DIMS(wheremask),
                    PyArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            goto fail;
        }

        /* Do the masked assignment with raw array iteration */
        if (raw_array_wheremasked_assign_scalar(
                PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                src_dtype, src_data,
                PyArray_DESCR(wheremask), PyArray_DATA(wheremask),
                wheremask_strides, casting) < 0) {
            goto fail;
        }
    }

    npy_free_workspace(tmp_src_data);
    return 0;

fail:
    npy_free_workspace(tmp_src_data);
    return -1;
}
