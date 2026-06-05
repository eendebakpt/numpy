# NumPy Core C API - Object Relationships (UML)

> C structures with their Python counterparts annotated.
> Edit the Mermaid source below or the companion `.puml` file for PlantUML.

```mermaid
classDiagram
    direction TB

    class PyArrayObject_fields {
        <<np.ndarray>>
        +PyObject_HEAD
        +char* data
        +int nd
        +npy_intp* dimensions
        +npy_intp* strides
        +PyObject* base
        +PyArray_Descr* descr
        +int flags
        +PyObject* mem_handler
    }

    class PyArray_Descr {
        <<np.dtype>>
        +PyObject_HEAD
        +PyTypeObject* typeobj
        +char kind
        +char type
        +char byteorder
        +int type_num
        +npy_uint64 flags
        +npy_intp elsize
        +npy_intp alignment
        +PyObject* metadata
    }

    class _PyArray_LegacyDescr {
        <<extends PyArray_Descr>>
        +_arr_descr* subarray
        +PyObject* fields
        +PyObject* names
        +NpyAuxData* c_metadata
    }

    class PyArray_DTypeMeta {
        <<type of np.dtype - metaclass>>
        +PyHeapTypeObject super
        +PyArray_Descr* singleton
        +int type_num
        +PyTypeObject* scalar_type
        +npy_uint64 flags
        +void* dt_slots
    }

    class PyUFuncObject {
        <<np.ufunc e.g. np.add>>
        +PyObject_HEAD
        +int nin
        +int nout
        +int nargs
        +int identity
        +PyUFuncGenericFunction* functions
        +void** data
        +int ntypes
        +char* name
        +char* types
        +int core_enabled
        +char* core_signature
        +void* _dispatch_cache
        +PyObject* _loops
    }

    class PyArrayMethodObject {
        <<internal loop method>>
        +PyObject_HEAD
        +char* name
        +int nin
        +int nout
        +NPY_CASTING casting
        +NPY_ARRAYMETHOD_FLAGS flags
        +void* static_data
        +resolve_descriptors()
        +get_strided_loop()
        +strided_loop()
        +contiguous_loop()
        +unaligned_strided_loop()
        +PyArrayMethodObject* wrapped_meth
        +PyArray_DTypeMeta** wrapped_dtypes
        +void* legacy_loop
    }

    class PyBoundArrayMethodObject {
        <<bound method + dtypes>>
        +PyObject_HEAD
        +PyArray_DTypeMeta** dtypes
        +PyArrayMethodObject* method
    }

    class PyArrayMethod_Context {
        <<runtime context>>
        +PyObject* caller
        +PyArrayMethodObject* method
        +PyArray_Descr** descriptors
    }

    class PyUFuncGenericFunction {
        <<typedef - legacy loop signature>>
        void callback(char** args, npy_intp* dimensions, npy_intp* strides, void* data)
    }

    class PyArrayMethod_StridedLoop {
        <<typedef - new loop signature>>
        int callback(PyArrayMethod_Context*, char**, npy_intp*, npy_intp*, NpyAuxData*)
    }

    class PyArray_ArrFuncs {
        <<legacy per-dtype operations>>
        +cast[]
        +getitem()
        +setitem()
        +copyswapn()
        +compare()
        +argmax()
        +argmin()
        +dotfunc()
        +fill()
        +sort[]
        +nonzero()
    }

    %% Relationships

    PyArrayObject_fields --> PyArray_Descr : descr

    PyArray_DTypeMeta ..> PyArray_Descr : metaclass\nNPY_DTYPE(descr)
    PyArray_DTypeMeta --> PyArray_Descr : singleton

    _PyArray_LegacyDescr --|> PyArray_Descr : extends

    PyUFuncObject o--> PyUFuncGenericFunction : functions[] (legacy)
    PyUFuncObject o--> PyArrayMethodObject : _loops registry

    PyArrayMethodObject --> PyArray_DTypeMeta : wrapped_dtypes
    PyArrayMethodObject --> PyUFuncGenericFunction : legacy_loop (wrapped)
    PyArrayMethodObject --> PyArrayMethodObject : wrapped_meth
    PyArrayMethodObject --> PyArrayMethod_StridedLoop : loop implementations

    PyBoundArrayMethodObject --> PyArrayMethodObject : method
    PyBoundArrayMethodObject --> PyArray_DTypeMeta : dtypes[]

    PyArrayMethod_Context --> PyArrayMethodObject : method
    PyArrayMethod_Context --> PyArray_Descr : descriptors[]
    PyArrayMethod_Context ..> PyUFuncObject : caller

    _PyArray_LegacyDescr ..> PyArray_ArrFuncs : f (legacy)
```

## Key Concepts

| C Structure | Python Equivalent | Role |
|---|---|---|
| `PyArrayObject_fields` | `np.ndarray` | N-dimensional array |
| `PyArray_Descr` | `np.dtype` | Data type descriptor (element type, size, byte order) |
| `PyArray_DTypeMeta` | `type(np.dtype(...))` | Metaclass for dtype instances (new DType system) |
| `PyUFuncObject` | `np.ufunc` (e.g. `np.add`) | Universal function with loop dispatch |
| `PyArrayMethodObject` | (internal) | Single typed implementation of a ufunc operation |
| `PyBoundArrayMethodObject` | (internal) | ArrayMethod bound to specific DType classes |
| `PyUFuncGenericFunction` | (typedef) | Legacy loop function pointer signature |
| `PyArrayMethod_StridedLoop` | (typedef) | New-style loop function pointer signature |
| `PyArray_ArrFuncs` | (legacy) | Per-dtype operation table (being phased out) |

## Dispatch Flow

1. **`PyUFuncObject`** receives a call (e.g. `np.add(a, b)`)
2. Resolves input dtypes and looks up the best **`PyArrayMethodObject`** from `_loops`
3. The **`PyArrayMethodObject`** resolves output descriptors via `resolve_descriptors()`
4. A concrete **`PyArrayMethod_StridedLoop`** (or legacy **`PyUFuncGenericFunction`**) is selected
5. The loop executes over array data, using **`PyArrayMethod_Context`** for runtime info

## Files

| Structure | Header |
|---|---|
| `PyArrayObject_fields`, `PyArray_Descr`, `PyArray_ArrFuncs` | `numpy/_core/include/numpy/ndarraytypes.h` |
| `PyUFuncObject`, `PyUFuncGenericFunction` | `numpy/_core/include/numpy/ufuncobject.h` |
| `PyArray_DTypeMeta` | `numpy/_core/include/numpy/dtype_api.h` |
| `PyArrayMethodObject` | `numpy/_core/src/multiarray/array_method.h` |
