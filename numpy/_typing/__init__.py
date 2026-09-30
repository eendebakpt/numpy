"""Private counterpart of ``numpy.typing``."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from ._array_like import (
        ArrayLike as ArrayLike,
        NDArray as NDArray,
        _ArrayLike as _ArrayLike,
        _ArrayLikeAnyString_co as _ArrayLikeAnyString_co,
        _ArrayLikeBool_co as _ArrayLikeBool_co,
        _ArrayLikeBytes_co as _ArrayLikeBytes_co,
        _ArrayLikeComplex128_co as _ArrayLikeComplex128_co,
        _ArrayLikeComplex_co as _ArrayLikeComplex_co,
        _ArrayLikeDT64_co as _ArrayLikeDT64_co,
        _ArrayLikeFloat64_co as _ArrayLikeFloat64_co,
        _ArrayLikeFloat_co as _ArrayLikeFloat_co,
        _ArrayLikeInt as _ArrayLikeInt,
        _ArrayLikeInt_co as _ArrayLikeInt_co,
        _ArrayLikeNumber_co as _ArrayLikeNumber_co,
        _ArrayLikeObject_co as _ArrayLikeObject_co,
        _ArrayLikeStr_co as _ArrayLikeStr_co,
        _ArrayLikeString_co as _ArrayLikeString_co,
        _ArrayLikeTD64_co as _ArrayLikeTD64_co,
        _ArrayLikeUInt_co as _ArrayLikeUInt_co,
        _ArrayLikeVoid_co as _ArrayLikeVoid_co,
        _SupportsArray as _SupportsArray,
        _SupportsArrayFunc as _SupportsArrayFunc,
    )

    #
    from ._char_codes import (
        _BoolCodes as _BoolCodes,
        _BytesCodes as _BytesCodes,
        _CharacterCodes as _CharacterCodes,
        _CLongDoubleCodes as _CLongDoubleCodes,
        _Complex64Codes as _Complex64Codes,
        _Complex128Codes as _Complex128Codes,
        _ComplexFloatingCodes as _ComplexFloatingCodes,
        _DT64Codes as _DT64Codes,
        _FlexibleCodes as _FlexibleCodes,
        _Float16Codes as _Float16Codes,
        _Float32Codes as _Float32Codes,
        _Float64Codes as _Float64Codes,
        _FloatingCodes as _FloatingCodes,
        _GenericCodes as _GenericCodes,
        _InexactCodes as _InexactCodes,
        _Int8Codes as _Int8Codes,
        _Int16Codes as _Int16Codes,
        _Int32Codes as _Int32Codes,
        _Int64Codes as _Int64Codes,
        _IntCCodes as _IntCCodes,
        _IntegerCodes as _IntegerCodes,
        _IntPCodes as _IntPCodes,
        _LongCodes as _LongCodes,
        _LongDoubleCodes as _LongDoubleCodes,
        _LongLongCodes as _LongLongCodes,
        _NumberCodes as _NumberCodes,
        _ObjectCodes as _ObjectCodes,
        _SignedIntegerCodes as _SignedIntegerCodes,
        _StrCodes as _StrCodes,
        _StringCodes as _StringCodes,
        _TD64Codes as _TD64Codes,
        _UInt8Codes as _UInt8Codes,
        _UInt16Codes as _UInt16Codes,
        _UInt32Codes as _UInt32Codes,
        _UInt64Codes as _UInt64Codes,
        _UIntCCodes as _UIntCCodes,
        _UIntPCodes as _UIntPCodes,
        _ULongCodes as _ULongCodes,
        _ULongLongCodes as _ULongLongCodes,
        _UnsignedIntegerCodes as _UnsignedIntegerCodes,
        _VoidCodes as _VoidCodes,
    )

    #
    from ._dtype_like import (
        DTypeLike as DTypeLike,
        _DTypeLike as _DTypeLike,
        _DTypeLikeBool as _DTypeLikeBool,
        _DTypeLikeBytes as _DTypeLikeBytes,
        _DTypeLikeComplex as _DTypeLikeComplex,
        _DTypeLikeComplex_co as _DTypeLikeComplex_co,
        _DTypeLikeDT64 as _DTypeLikeDT64,
        _DTypeLikeFloat as _DTypeLikeFloat,
        _DTypeLikeInt as _DTypeLikeInt,
        _DTypeLikeObject as _DTypeLikeObject,
        _DTypeLikeStr as _DTypeLikeStr,
        _DTypeLikeTD64 as _DTypeLikeTD64,
        _DTypeLikeUInt as _DTypeLikeUInt,
        _DTypeLikeVoid as _DTypeLikeVoid,
        _HasDType as _HasDType,
        _SupportsDType as _SupportsDType,
        _VoidDTypeLike as _VoidDTypeLike,
    )

    #
    from ._nbit import (
        _NBitByte as _NBitByte,
        _NBitDouble as _NBitDouble,
        _NBitHalf as _NBitHalf,
        _NBitInt as _NBitInt,
        _NBitIntC as _NBitIntC,
        _NBitIntP as _NBitIntP,
        _NBitLong as _NBitLong,
        _NBitLongDouble as _NBitLongDouble,
        _NBitLongLong as _NBitLongLong,
        _NBitShort as _NBitShort,
        _NBitSingle as _NBitSingle,
    )

    #
    from ._nbit_base import (  # type: ignore[deprecated]
        NBitBase as NBitBase,  # pyright: ignore[reportDeprecated]
        _8Bit as _8Bit,
        _16Bit as _16Bit,
        _32Bit as _32Bit,
        _64Bit as _64Bit,
        _96Bit as _96Bit,
        _128Bit as _128Bit,
    )

    #
    from ._nested_sequence import _NestedSequence as _NestedSequence

    #
    from ._scalars import (
        _BoolLike_co as _BoolLike_co,
        _CharLike_co as _CharLike_co,
        _ComplexLike_co as _ComplexLike_co,
        _FloatLike_co as _FloatLike_co,
        _IntLike_co as _IntLike_co,
        _NumberLike_co as _NumberLike_co,
        _ScalarLike_co as _ScalarLike_co,
        _TD64Like_co as _TD64Like_co,
        _UIntLike_co as _UIntLike_co,
        _VoidLike_co as _VoidLike_co,
    )

    #
    from ._shape import (
        _AnyShape as _AnyShape,
        _Shape as _Shape,
        _ShapeLike as _ShapeLike,
    )
else:
    # At runtime, the submodules are imported on first use of one of their
    # names (PEP 562).  This keeps ``import numpy`` cheap: only
    # ``numpy.linalg`` needs ``NDArray`` at runtime, and nothing else here.
    # NOTE: keep in sync with the imports above (checked by
    # numpy/typing/tests/test_runtime.py).
    _submodule_names = {
        "_array_like": (
            "ArrayLike",
            "NDArray",
            "_ArrayLike",
            "_ArrayLikeAnyString_co",
            "_ArrayLikeBool_co",
            "_ArrayLikeBytes_co",
            "_ArrayLikeComplex128_co",
            "_ArrayLikeComplex_co",
            "_ArrayLikeDT64_co",
            "_ArrayLikeFloat64_co",
            "_ArrayLikeFloat_co",
            "_ArrayLikeInt",
            "_ArrayLikeInt_co",
            "_ArrayLikeNumber_co",
            "_ArrayLikeObject_co",
            "_ArrayLikeStr_co",
            "_ArrayLikeString_co",
            "_ArrayLikeTD64_co",
            "_ArrayLikeUInt_co",
            "_ArrayLikeVoid_co",
            "_SupportsArray",
            "_SupportsArrayFunc",
        ),
        "_char_codes": (
            "_BoolCodes",
            "_BytesCodes",
            "_CharacterCodes",
            "_CLongDoubleCodes",
            "_Complex64Codes",
            "_Complex128Codes",
            "_ComplexFloatingCodes",
            "_DT64Codes",
            "_FlexibleCodes",
            "_Float16Codes",
            "_Float32Codes",
            "_Float64Codes",
            "_FloatingCodes",
            "_GenericCodes",
            "_InexactCodes",
            "_Int8Codes",
            "_Int16Codes",
            "_Int32Codes",
            "_Int64Codes",
            "_IntCCodes",
            "_IntegerCodes",
            "_IntPCodes",
            "_LongCodes",
            "_LongDoubleCodes",
            "_LongLongCodes",
            "_NumberCodes",
            "_ObjectCodes",
            "_SignedIntegerCodes",
            "_StrCodes",
            "_StringCodes",
            "_TD64Codes",
            "_UInt8Codes",
            "_UInt16Codes",
            "_UInt32Codes",
            "_UInt64Codes",
            "_UIntCCodes",
            "_UIntPCodes",
            "_ULongCodes",
            "_ULongLongCodes",
            "_UnsignedIntegerCodes",
            "_VoidCodes",
        ),
        "_dtype_like": (
            "DTypeLike",
            "_DTypeLike",
            "_DTypeLikeBool",
            "_DTypeLikeBytes",
            "_DTypeLikeComplex",
            "_DTypeLikeComplex_co",
            "_DTypeLikeDT64",
            "_DTypeLikeFloat",
            "_DTypeLikeInt",
            "_DTypeLikeObject",
            "_DTypeLikeStr",
            "_DTypeLikeTD64",
            "_DTypeLikeUInt",
            "_DTypeLikeVoid",
            "_HasDType",
            "_SupportsDType",
            "_VoidDTypeLike",
        ),
        "_nbit": (
            "_NBitByte",
            "_NBitDouble",
            "_NBitHalf",
            "_NBitInt",
            "_NBitIntC",
            "_NBitIntP",
            "_NBitLong",
            "_NBitLongDouble",
            "_NBitLongLong",
            "_NBitShort",
            "_NBitSingle",
        ),
        "_nbit_base": (
            "NBitBase",
            "_8Bit",
            "_16Bit",
            "_32Bit",
            "_64Bit",
            "_96Bit",
            "_128Bit",
        ),
        "_nested_sequence": (
            "_NestedSequence",
        ),
        "_scalars": (
            "_BoolLike_co",
            "_CharLike_co",
            "_ComplexLike_co",
            "_FloatLike_co",
            "_IntLike_co",
            "_NumberLike_co",
            "_ScalarLike_co",
            "_TD64Like_co",
            "_UIntLike_co",
            "_VoidLike_co",
        ),
        "_shape": (
            "_AnyShape",
            "_Shape",
            "_ShapeLike",
        ),
    }
    _name_to_submodule = {
        name: submodule
        for submodule, names in _submodule_names.items()
        for name in names
    }
    del _submodule_names

    def __getattr__(name):
        try:
            submodule = _name_to_submodule[name]
        except KeyError:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}") from None
        import importlib
        module = importlib.import_module(f"{__name__}.{submodule}")
        value = getattr(module, name)
        globals()[name] = value
        return value
