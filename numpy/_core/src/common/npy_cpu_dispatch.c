#include "npy_cpu_dispatch.h"
#include "numpy/ndarraytypes.h"
#include "npy_static_data.h"
#include "module_state.h"

/*
 * Dispatch information is only recorded here (a few thousand entries at
 * import time); the `__cpu_targets_info__` dictionary is built from these
 * records on first access (see `npy_cpu_dispatch_targets_info`).
 */
typedef struct npy_cpu_dispatch_record {
    const char *fname;      /* string literal at the trace site */
    const char *current;    /* string literals from NPY_CPU_DISPATCH_INFO() */
    const char *available;
    char signature[];       /* copied: the trace site may pass stack memory */
} npy_cpu_dispatch_record;

NPY_VISIBILITY_HIDDEN int
npy_cpu_dispatch_tracer_init(PyObject *mod)
{
    multiarray_umath_state *state = get_module_state(mod);
    if (state->static_cdata.cpu_dispatch_records != NULL) {
        PyErr_Format(PyExc_RuntimeError, "CPU dispatcher tracer already initialized");
        return -1;
    }
    state->static_cdata.n_cpu_dispatch_records = 0;
    state->static_cdata.cpu_dispatch_records_capacity = 0;
    return 0;
}

NPY_VISIBILITY_HIDDEN void
npy_cpu_dispatch_trace(const char *fname, const char *signature,
                       const char **dispatch_info)
{
    npy_static_cdata_struct *cdata = &_npy_module_state->static_cdata;
    if (cdata->n_cpu_dispatch_records == cdata->cpu_dispatch_records_capacity) {
        Py_ssize_t capacity = cdata->cpu_dispatch_records_capacity;
        capacity = capacity == 0 ? 1024 : 2 * capacity;
        npy_cpu_dispatch_record **records = PyMem_RawRealloc(
                cdata->cpu_dispatch_records, capacity * sizeof(*records));
        if (records == NULL) {
            return;  /* tracing is best effort */
        }
        cdata->cpu_dispatch_records = records;
        cdata->cpu_dispatch_records_capacity = capacity;
    }
    size_t siglen = strlen(signature);
    npy_cpu_dispatch_record *record = PyMem_RawMalloc(sizeof(*record) + siglen + 1);
    if (record == NULL) {
        return;
    }
    record->fname = fname;
    record->current = dispatch_info[0];
    record->available = dispatch_info[1];
    memcpy(record->signature, signature, siglen + 1);
    cdata->cpu_dispatch_records[cdata->n_cpu_dispatch_records++] = record;
}

NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_dispatch_targets_info(void)
{
    npy_static_cdata_struct *cdata = &_npy_module_state->static_cdata;
    PyObject *reg_dict = PyDict_New();
    if (reg_dict == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < cdata->n_cpu_dispatch_records; i++) {
        npy_cpu_dispatch_record *record = cdata->cpu_dispatch_records[i];
        PyObject *func_dict = PyDict_GetItemString(reg_dict, record->fname); // noqa: borrowed-ref OK
        if (func_dict == NULL) {
            func_dict = PyDict_New();
            if (func_dict == NULL) {
                goto fail;
            }
            int err = PyDict_SetItemString(reg_dict, record->fname, func_dict);
            Py_DECREF(func_dict);
            if (err != 0) {
                goto fail;
            }
        }
        // target info for each signature
        PyObject *sig_dict = Py_BuildValue("{s:s, s:s}",
                "current", record->current, "available", record->available);
        if (sig_dict == NULL) {
            goto fail;
        }
        int err = PyDict_SetItemString(func_dict, record->signature, sig_dict);
        Py_DECREF(sig_dict);
        if (err != 0) {
            goto fail;
        }
    }
    return reg_dict;

  fail:
    Py_DECREF(reg_dict);
    return NULL;
}
