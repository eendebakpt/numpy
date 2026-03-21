/*
 * Size-based freelists for NumPy scalar objects.
 *
 * Inspired by CPython's freelist infrastructure
 * (cpython/Include/internal/pycore_freelist.h).
 *
 * PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
 * Copyright (c) 2001-2025 Python Software Foundation.
 * All Rights Reserved.
 *
 * SPDX-License-Identifier: PSF-2.0
 *
 * Design:
 *   - Thread safety is achieved using thread-local storage (NPY_TLS).
 *     Each thread has its own set of freelists, so no locking is needed.
 *   - Freelists are organized by allocation size (tp_basicsize), not by
 *     type. This allows a single freelist to serve all scalar types of the
 *     same size (e.g., int64, uint64, float64, complex64 all share the
 *     basicsize=40 freelist).
 *   - Dead objects are linked through their first sizeof(void*) bytes
 *     (overlapping ob_refcnt / ob_tid), following CPython's convention.
 */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_NPY_FREELIST_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NPY_FREELIST_H_

#include "numpy/npy_common.h"
#include <Python.h>

/*
 * NumPy numeric scalar tp_basicsize values (platform-independent):
 *   40 — bool, int8/16/32/64, uint8/16/32/64, float16/32/64, complex64
 *   48 — longdouble, complex128
 *   64 — clongdouble
 *
 * We use three freelists, one per size class.
 */
#define NPY_SCALAR_FREELIST_SIZE 40
#define NPY_NUM_SCALAR_FREELISTS 3

typedef struct {
    void *head;      /* singly-linked list of free objects */
    int size;        /* number of items on this freelist */
} npy_scalar_freelist;

/* Allocation sizes served by each freelist slot. */
static const int _npy_freelist_sizes[NPY_NUM_SCALAR_FREELISTS] = {40, 48, 64};

static NPY_TLS npy_scalar_freelist
        _npy_scalar_freelists[NPY_NUM_SCALAR_FREELISTS] = {{NULL, 0}, {NULL, 0}, {NULL, 0}};

/*
 * Map an allocation size to a freelist index, or -1 if not served.
 */
static inline int
_npy_freelist_index(int basicsize)
{
    for (int i = 0; i < NPY_NUM_SCALAR_FREELISTS; i++) {
        if (basicsize == _npy_freelist_sizes[i]) {
            return i;
        }
    }
    return -1;
}

/*
 * Try to pop an object from the freelist for the given size.
 * Returns a raw pointer with undefined contents (caller must
 * call PyObject_Init), or NULL if the freelist is empty.
 */
static inline void *
npy_scalar_freelist_pop(int basicsize)
{
    int idx = _npy_freelist_index(basicsize);
    if (idx < 0) {
        return NULL;
    }
    npy_scalar_freelist *fl = &_npy_scalar_freelists[idx];
    void *obj = fl->head;
    if (obj != NULL) {
        fl->head = *(void **)obj;
        fl->size--;
    }
    return obj;
}

/*
 * Try to push a dead object onto the freelist for the given size.
 * Returns 1 if the object was recycled, 0 if the freelist is full
 * (caller should fall back to PyObject_Free).
 */
static inline int
npy_scalar_freelist_push(int basicsize, void *obj)
{
    int idx = _npy_freelist_index(basicsize);
    if (idx < 0 || _npy_scalar_freelists[idx].size >= NPY_SCALAR_FREELIST_SIZE) {
        return 0;
    }
    npy_scalar_freelist *fl = &_npy_scalar_freelists[idx];
    *(void **)obj = fl->head;
    fl->head = obj;
    fl->size++;
    return 1;
}

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NPY_FREELIST_H_ */
