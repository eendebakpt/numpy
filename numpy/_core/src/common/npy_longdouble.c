#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"


/*
 * Heavily derived from PyLong_FromDouble
 * Notably, we can't set the digits directly, so have to shift and or instead.
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_longdouble_to_PyLong(npy_longdouble ldval)
{
    PyObject *v;
    PyObject *l_chunk_size;
    /*
     * number of bits to extract at a time. CPython uses 30, but that's because
     * it's tied to the internal long representation
     */
    const int chunk_size = NPY_BITSOF_LONGLONG;
    npy_longdouble frac;
    int i, ndig, expo, neg;
    neg = 0;

    if (npy_isinf(ldval)) {
        PyErr_SetString(PyExc_OverflowError,
                        "cannot convert longdouble infinity to integer");
        return NULL;
    }
    if (npy_isnan(ldval)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot convert longdouble NaN to integer");
        return NULL;
    }
    if (ldval < 0.0) {
        neg = 1;
        ldval = -ldval;
    }
    frac = npy_frexpl(ldval, &expo); /* ldval = frac*2**expo; 0.0 <= frac < 1.0 */
    v = PyLong_FromLong(0L);
    if (v == NULL)
        return NULL;
    if (expo <= 0)
        return v;

    ndig = (expo-1) / chunk_size + 1;

    l_chunk_size = PyLong_FromLong(chunk_size);
    if (l_chunk_size == NULL) {
        Py_DECREF(v);
        return NULL;
    }

    /* Get the MSBs of the integral part of the float */
    frac = npy_ldexpl(frac, (expo-1) % chunk_size + 1);
    for (i = ndig; --i >= 0; ) {
        npy_ulonglong chunk = (npy_ulonglong)frac;
        PyObject *l_chunk;
        /* v = v << chunk_size */
        Py_SETREF(v, PyNumber_Lshift(v, l_chunk_size));
        if (v == NULL) {
            goto done;
        }
        l_chunk = PyLong_FromUnsignedLongLong(chunk);
        if (l_chunk == NULL) {
            Py_DECREF(v);
            v = NULL;
            goto done;
        }
        /* v = v | chunk */
        Py_SETREF(v, PyNumber_Or(v, l_chunk));
        Py_DECREF(l_chunk);
        if (v == NULL) {
            goto done;
        }

        /* Remove the msbs, and repeat */
        frac = frac - (npy_longdouble) chunk;
        frac = npy_ldexpl(frac, chunk_size);
    }

    /* v = -v */
    if (neg) {
        Py_SETREF(v, PyNumber_Negative(v));
        if (v == NULL) {
            goto done;
        }
    }

done:
    Py_DECREF(l_chunk_size);
    return v;
}

#include <float.h>

/*
 * Convert a Python int to long double with a single, correct rounding.
 *
 * Small ints are converted directly.  For larger ones the top
 * LDBL_MANT_DIG + 2 bits are extracted, a sticky bit records whether any
 * lower bit was set, and the result is scaled back with ldexp.  Only the
 * final addition rounds, so round-half-even is applied to the exact value.
 */
NPY_VISIBILITY_HIDDEN npy_longdouble
npy_longdouble_from_PyLong(PyObject *long_obj)
{
    int overflow;
    long long ll = PyLong_AsLongLongAndOverflow(long_obj, &overflow);
    if (overflow == 0) {
        if (ll == -1 && PyErr_Occurred()) {
            return -1;
        }
        return (npy_longdouble)ll;
    }

    npy_longdouble result = -1;
    PyObject *absval = NULL, *nbits_obj = NULL, *shift_obj = NULL;
    PyObject *top = NULL, *back = NULL, *hi_obj = NULL, *sixtyfour = NULL;

    absval = PyNumber_Absolute(long_obj);
    if (absval == NULL) {
        goto done;
    }
    nbits_obj = PyObject_CallMethod(absval, "bit_length", NULL);
    if (nbits_obj == NULL) {
        goto done;
    }
    long nbits = PyLong_AsLong(nbits_obj);
    if (nbits == -1 && PyErr_Occurred()) {
        goto done;
    }
    long shift = nbits - (LDBL_MANT_DIG + 2);
    int sticky = 0;
    if (shift > 0) {
        shift_obj = PyLong_FromLong(shift);
        if (shift_obj == NULL) {
            goto done;
        }
        top = PyNumber_Rshift(absval, shift_obj);
        if (top == NULL) {
            goto done;
        }
        back = PyNumber_Lshift(top, shift_obj);
        if (back == NULL) {
            goto done;
        }
        sticky = PyObject_RichCompareBool(back, absval, Py_NE);
        if (sticky < 0) {
            goto done;
        }
    }
    else {
        shift = 0;
        top = absval;
        Py_INCREF(top);
    }
    /* top has at most LDBL_MANT_DIG + 2 <= 115 bits: split into two halves */
    sixtyfour = PyLong_FromLong(64);
    if (sixtyfour == NULL) {
        goto done;
    }
    hi_obj = PyNumber_Rshift(top, sixtyfour);
    if (hi_obj == NULL) {
        goto done;
    }
    unsigned long long hi = PyLong_AsUnsignedLongLong(hi_obj);
    if (hi == (unsigned long long)-1 && PyErr_Occurred()) {
        goto done;
    }
    unsigned long long lo = PyLong_AsUnsignedLongLongMask(top) | (unsigned long long)sticky;

    /* hi and lo are exact; this addition is the only rounding step */
    result = npy_ldexpl((npy_longdouble)hi, 64) + (npy_longdouble)lo;
    result = npy_ldexpl(result, (int)shift);
    if (npy_isinf(result)) {
        PyErr_SetString(PyExc_OverflowError,
                        "int too large to convert to longdouble");
        result = -1;
        goto done;
    }
    if (overflow < 0) {
        result = -result;
    }
done:
    Py_XDECREF(absval);
    Py_XDECREF(nbits_obj);
    Py_XDECREF(shift_obj);
    Py_XDECREF(top);
    Py_XDECREF(back);
    Py_XDECREF(hi_obj);
    Py_XDECREF(sixtyfour);
    return result;
}
