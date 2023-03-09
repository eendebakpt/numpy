#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

NPY_VISIBILITY_HIDDEN extern int npy_numpy2_behavior;

NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_err_ignore;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_err_warn;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_err_raise;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_err_call;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_err_print;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_err_log;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_err_divide;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_err_over;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_err_under;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_err_invalid;

NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_current_allocator;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_array;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_array_function;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_array_struct;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_array_priority;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_array_interface;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_array_wrap;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_array_finalize;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_implementation;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_axis1;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_axis2;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_like;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_numpy;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */
