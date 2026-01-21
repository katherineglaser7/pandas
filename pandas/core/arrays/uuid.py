"""
Extension array for UUID data.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    ClassVar,
    Self,
)
from uuid import UUID

import numpy as np

from pandas._libs import (
    missing as libmissing,
)
from pandas.util._decorators import set_module

from pandas.core.dtypes.base import (
    ExtensionDtype,
    register_extension_dtype,
)
from pandas.core.dtypes.common import (
    is_list_like,
    pandas_dtype,
)

from pandas.core.arrays.base import ExtensionArray
from pandas.core.indexers import check_array_indexer

if TYPE_CHECKING:
    import pyarrow

    from pandas._typing import (
        Dtype,
        npt,
        type_t,
    )


@register_extension_dtype
@set_module("pandas")
class UUIDDtype(ExtensionDtype):
    """
    Extension dtype for UUID data.

    This dtype represents UUID (Universally Unique Identifier) values,
    providing first-class support for UUID data in pandas.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    arrays.UUIDArray : Array of UUID data.
    StringDtype : Extension dtype for string data.

    Examples
    --------
    >>> pd.UUIDDtype()
    UUIDDtype

    >>> pd.array(
    ...     ["a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"], dtype=pd.UUIDDtype()
    ... )
    <UUIDArray>
    [UUID('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11')]
    Length: 1, dtype: uuid
    """

    name: ClassVar[str] = "uuid"
    _metadata: tuple[str, ...] = ()

    @property
    def type(self) -> type[UUID]:
        return UUID

    @property
    def kind(self) -> str:
        return "O"

    @property
    def na_value(self) -> libmissing.NAType:
        return libmissing.NA

    def construct_array_type(self) -> type_t[UUIDArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return UUIDArray

    @classmethod
    def construct_from_string(cls, string: str) -> Self:
        """
        Construct a UUIDDtype from a string.

        Parameters
        ----------
        string : str
            The string to construct from. Must be "uuid".

        Returns
        -------
        UUIDDtype

        Raises
        ------
        TypeError
            If the string is not "uuid".
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if string != cls.name:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        return cls()

    def __repr__(self) -> str:
        return "UUIDDtype"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other == self.name
        return isinstance(other, type(self))

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> UUIDArray:
        """
        Construct UUIDArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        if isinstance(array, pyarrow.ChunkedArray):
            chunks = array.chunks
        else:
            chunks = [array]

        results = []
        for arr in chunks:
            # Convert to Python objects
            arr_np = arr.to_numpy(zero_copy_only=False)
            uuid_arr = _coerce_to_uuid_array(arr_np)
            results.append(uuid_arr)

        if not results:
            data = np.array([], dtype=object)
        else:
            data = np.concatenate(results)

        return UUIDArray._simple_new(data)


def _coerce_to_uuid_array(
    values, *, copy: bool = False
) -> np.ndarray:
    """
    Coerce values to a numpy array of UUID objects.

    Parameters
    ----------
    values : array-like
    copy : bool, default False

    Returns
    -------
    np.ndarray
        Array of UUID objects with dtype=object
    """
    if isinstance(values, UUIDArray):
        if copy:
            return values._ndarray.copy()
        return values._ndarray

    if isinstance(values, np.ndarray) and values.dtype == object:
        result = np.empty(len(values), dtype=object)
        for i, val in enumerate(values):
            is_na = (
                val is None
                or (isinstance(val, float) and np.isnan(val))
                or val is libmissing.NA
            )
            if is_na:
                result[i] = libmissing.NA
            elif isinstance(val, UUID):
                result[i] = val
            elif isinstance(val, str):
                try:
                    result[i] = UUID(val)
                except ValueError as err:
                    raise ValueError(f"Cannot convert '{val}' to UUID") from err
            elif isinstance(val, bytes):
                try:
                    result[i] = UUID(bytes=val)
                except ValueError as err:
                    raise ValueError("Cannot convert bytes to UUID") from err
            else:
                raise TypeError(f"Cannot convert {type(val).__name__} to UUID")
        return result

    # Handle other iterables
    if not isinstance(values, (list, np.ndarray)):
        values_list = list(values)
    else:
        values_list = values
    result = np.empty(len(values_list), dtype=object)
    for i, val in enumerate(values_list):
        is_na = (
            val is None
            or (isinstance(val, float) and np.isnan(val))
            or val is libmissing.NA
        )
        if is_na:
            result[i] = libmissing.NA
        elif isinstance(val, UUID):
            result[i] = val
        elif isinstance(val, str):
            try:
                result[i] = UUID(val)
            except ValueError as err:
                raise ValueError(f"Cannot convert '{val}' to UUID") from err
        elif isinstance(val, bytes):
            try:
                result[i] = UUID(bytes=val)
            except ValueError as err:
                raise ValueError("Cannot convert bytes to UUID") from err
        else:
            raise TypeError(f"Cannot convert {type(val).__name__} to UUID")

    return result


@set_module("pandas.arrays")
class UUIDArray(ExtensionArray):
    """
    Array of UUID (Universally Unique Identifier) data.

    This is a pandas Extension array for UUID data, backed by a numpy
    array of UUID objects.

    To construct a UUIDArray from generic array-like input, use
    :func:`pandas.array` specifying ``dtype="uuid"`` (see examples below).

    Parameters
    ----------
    values : array-like
        The UUID values. Can be UUID objects, strings, or bytes.
    copy : bool, default False
        Whether to copy the underlying data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    UUIDArray

    See Also
    --------
    array : Create an array from data with the appropriate dtype.
    UUIDDtype : Extension dtype for UUID data.

    Examples
    --------
    Create a UUIDArray with :func:`pandas.array`:

    >>> pd.array(["a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"], dtype="uuid")
    <UUIDArray>
    [UUID('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11')]
    Length: 1, dtype: uuid
    """

    _dtype = UUIDDtype()

    def __init__(
        self, values, *, copy: bool = False
    ) -> None:
        self._ndarray = _coerce_to_uuid_array(values, copy=copy)

    @classmethod
    def _simple_new(cls, values: np.ndarray) -> Self:
        """
        Construct a UUIDArray from a numpy array without validation.
        """
        result = cls.__new__(cls)
        result._ndarray = values
        return result

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        """
        Construct a UUIDArray from a sequence of scalars.
        """
        return cls(scalars, copy=copy)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ) -> Self:
        """
        Construct a UUIDArray from a sequence of strings.
        """
        return cls(strings, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original) -> Self:
        """
        Reconstruct a UUIDArray after factorization.
        """
        return cls(values)

    def __getitem__(self, key):
        key = check_array_indexer(self, key)
        if isinstance(key, int):
            return self._ndarray[key]
        return type(self)._simple_new(self._ndarray[key])

    def __setitem__(self, key, value) -> None:
        key = check_array_indexer(self, key)
        if is_list_like(value):
            value = _coerce_to_uuid_array(value)
        elif value is not libmissing.NA and not isinstance(value, UUID):
            if isinstance(value, str):
                value = UUID(value)
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                value = libmissing.NA
            else:
                raise TypeError(f"Cannot convert {type(value).__name__} to UUID")
        self._ndarray[key] = value

    def __len__(self) -> int:
        return len(self._ndarray)

    def __eq__(self, other) -> np.ndarray:
        if isinstance(other, UUIDArray):
            other = other._ndarray
        elif isinstance(other, str):
            try:
                other = UUID(other)
            except ValueError:
                return np.zeros(len(self), dtype=bool)
        elif isinstance(other, UUID):
            pass
        elif other is libmissing.NA or other is None:
            # Comparison with NA returns NA for each element
            result = np.empty(len(self), dtype=object)
            result[:] = libmissing.NA
            return result
        else:
            return NotImplemented

        result = np.zeros(len(self), dtype=bool)
        mask = self.isna()
        if isinstance(other, np.ndarray):
            other_mask = np.array([x is libmissing.NA for x in other], dtype=bool)
            valid = ~mask & ~other_mask
            result[valid] = self._ndarray[valid] == other[valid]
        else:
            result[~mask] = self._ndarray[~mask] == other
        return result

    def __ne__(self, other) -> np.ndarray:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        if isinstance(result[0], type(libmissing.NA)):
            return result
        return ~result

    @property
    def dtype(self) -> UUIDDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        # Each UUID is 128 bits = 16 bytes, plus object overhead
        return self._ndarray.nbytes

    def isna(self) -> np.ndarray:
        return np.array([x is libmissing.NA for x in self._ndarray], dtype=bool)

    def take(
        self,
        indices,
        *,
        allow_fill: bool = False,
        fill_value=None,
    ) -> Self:
        from pandas.api.extensions import take

        if fill_value is None:
            fill_value = self.dtype.na_value

        result = take(
            self._ndarray, indices, allow_fill=allow_fill, fill_value=fill_value
        )
        return type(self)._simple_new(result)

    def copy(self) -> Self:
        return type(self)._simple_new(self._ndarray.copy())

    @classmethod
    def _concat_same_type(cls, to_concat) -> Self:
        return cls._simple_new(np.concatenate([x._ndarray for x in to_concat]))

    def _values_for_factorize(self) -> tuple[np.ndarray, object]:
        return self._ndarray, libmissing.NA

    def _values_for_argsort(self) -> np.ndarray:
        # Convert UUIDs to strings for sorting
        return np.array(
            [str(x) if x is not libmissing.NA else "" for x in self._ndarray]
        )

    def unique(self) -> Self:
        from pandas import unique as pd_unique

        return type(self)._simple_new(pd_unique(self._ndarray))

    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, UUIDDtype):
            if copy:
                return self.copy()
            return self
        elif dtype == "object":
            return self._ndarray.copy() if copy else self._ndarray
        elif dtype == "string" or (hasattr(dtype, "name") and dtype.name == "string"):
            result = [
                str(x) if x is not libmissing.NA else libmissing.NA
                for x in self._ndarray
            ]
            return np.array(result, dtype=object)
        return super().astype(dtype, copy=copy)

    def _formatter(self, boxed: bool = False):
        def fmt(x):
            if x is libmissing.NA:
                return "<NA>"
            return repr(x)
        return fmt

    def __repr__(self) -> str:
        data = list(self._ndarray)
        return f"<UUIDArray>\n{data}\nLength: {len(self)}, dtype: {self.dtype.name}"

    def _reduce(self, name: str, *, skipna: bool = True, **kwargs):
        if name in ("min", "max"):
            # Convert to strings for comparison
            mask = self.isna()
            if skipna:
                valid = self._ndarray[~mask]
            else:
                if mask.any():
                    return libmissing.NA
                valid = self._ndarray

            if len(valid) == 0:
                return libmissing.NA

            str_values = np.array([str(x) for x in valid])
            if name == "min":
                idx = str_values.argmin()
            else:
                idx = str_values.argmax()
            return valid[idx]

        raise TypeError(f"Cannot perform reduction '{name}' with UUID dtype")

    def __hash__(self):
        raise TypeError(f"unhashable type: '{type(self).__name__}'")

    def _hash_pandas_object(
        self,
        *,
        encoding: str,
        hash_key: str,
        categorize: bool,
    ) -> npt.NDArray[np.uint64]:
        from pandas.core.util.hashing import hash_array

        # Convert UUIDs to their string representation for hashing
        values = np.array(
            [str(x) if x is not libmissing.NA else None for x in self._ndarray],
            dtype=object,
        )
        return hash_array(
            values, encoding=encoding, hash_key=hash_key, categorize=categorize
        )
