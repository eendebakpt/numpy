/*
 * Size-based freelists for NumPy scalar objects.
 *
 * Inspired by CPython's freelist infrastructure
 * (cpython/Include/internal/pycore_freelist.h).
 *
 * Design:
 *   - Freelists are per-thread, accessed via PyThread_tss_get() which
 *     uses pthread_getspecific() — much cheaper than __tls_get_addr
 *     for dlopen'd extension modules.
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

/*
 * Freelist slots are indexed by (basicsize - 40) / 8:
 *   basicsize 40 -> index 0  (bool, int8..64, uint8..64, float16..64, complex64)
 *   basicsize 48 -> index 1  (longdouble, complex128)
 *   basicsize 56 -> index 2  (unused, kept as padding)
 *   basicsize 64 -> index 3  (clongdouble)
 */
#define NPY_SCALAR_FREELIST_MINSIZE 40
#define NPY_SCALAR_FREELIST_MAXSIZE 64
#define NPY_NUM_SCALAR_FREELISTS \
    ((NPY_SCALAR_FREELIST_MAXSIZE - NPY_SCALAR_FREELIST_MINSIZE) / 8 + 1)

typedef struct {
    void *head;      /* singly-linked list of free objects */
    int size;        /* number of items on this freelist */
} npy_scalar_freelist;

typedef struct {
    npy_scalar_freelist freelists[NPY_NUM_SCALAR_FREELISTS];
} npy_scalar_freelists;

/*
 * Thread-specific key for per-thread freelists.
 * Must be initialized once via npy_freelist_init().
 */
extern Py_tss_t _npy_freelist_tss_key;

/*
 * Initialize the TSS key. Call once during module init.
 * Returns 0 on success, -1 on failure.
 */
static inline int
npy_freelist_init(void)
{
    if (PyThread_tss_is_created(&_npy_freelist_tss_key)) {
        return 0;
    }
    return PyThread_tss_create(&_npy_freelist_tss_key);
}

static inline npy_scalar_freelists *
_npy_get_freelists(void)
{
    void *ptr = PyThread_tss_get(&_npy_freelist_tss_key);
    if (NPY_LIKELY(ptr != NULL)) {
        return (npy_scalar_freelists *)ptr;
    }
    npy_scalar_freelists *fl = (npy_scalar_freelists *)PyMem_RawCalloc(
        1, sizeof(npy_scalar_freelists));
    if (fl != NULL) {
        PyThread_tss_set(&_npy_freelist_tss_key, fl);
    }
    return fl;
}

/*
 * Map an allocation size to a freelist index, or -1 if not served.
 */
static inline int
_npy_freelist_index(int basicsize)
{
    assert(basicsize >= NPY_SCALAR_FREELIST_MINSIZE);
    assert(basicsize % 8 == 0);
    unsigned int idx = (unsigned int)(basicsize - NPY_SCALAR_FREELIST_MINSIZE) / 8;
    if (idx >= NPY_NUM_SCALAR_FREELISTS) {
        return -1;
    }
    return (int)idx;
}

/*
 * Try to pop an object from the freelist for the given size.
 * Returns a raw pointer (caller must initialize), or NULL if empty.
 */
static inline void *
npy_scalar_freelist_pop(int basicsize)
{
    int idx = _npy_freelist_index(basicsize);
    if (idx < 0) {
        return NULL;
    }
    npy_scalar_freelists *fls = _npy_get_freelists();
    if (NPY_UNLIKELY(fls == NULL)) {
        return NULL;
    }
    npy_scalar_freelist *fl = &fls->freelists[idx];
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
    if (idx < 0) {
        return 0;
    }
    npy_scalar_freelists *fls = _npy_get_freelists();
    if (NPY_UNLIKELY(fls == NULL)) {
        return 0;
    }
    npy_scalar_freelist *fl = &fls->freelists[idx];
    if (fl->size >= NPY_SCALAR_FREELIST_SIZE) {
        return 0;
    }
    *(void **)obj = fl->head;
    fl->head = obj;
    fl->size++;
    return 1;
}

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NPY_FREELIST_H_ */
