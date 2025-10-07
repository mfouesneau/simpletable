"""
DictDataFrame, a simplistic column based dataframe

The :class:`DataFrame` container allows easier manipulations of the data but is
basically deriving dictionary objects. In particular it allows easy conversions
to many common dataframe containers: `numpy.recarray`, `pandas.DataFrame`,
`dask.DataFrame`, `astropy.Table`, `xarray.Dataset`, `vaex.DataSetArrays`.

.. notes::

    * requirements: numpy
    * conversion to other formats require the appropriate library.

:author: Morgan Fouesneau
"""

import numpy as np
import numpy.typing as npt
import sys
import itertools
import operator
from typing import (
    Iterable,
    Dict,
    Hashable,
    Optional,
    Tuple,
    List,
    Generator,
    Iterator,
    Callable,
    Sequence,
    Any,
    ItemsView,
)

from . import convert
from .helpers import pretty_size_print

iteritems = operator.methodcaller("items")
itervalues = operator.methodcaller("values")
basestring = (str, bytes)

try:
    from .plotter import Plotter  # type: ignore
except ImportError:
    Plotter = None


__all__ = ["DictDataFrame"]


class DictDataFrame(dict):
    """
    A simple-ish dictionary like structure allowing usage as array on non
    constant multi-dimensional column data.

    It initializes like a normal dictionary and can be used as such.
    A few divergence points though: some default methods such as :func:`len`
    may refer to the lines and not the columns as a normal dictionary would.

    This data object implements also array slicing, shape, dtypes and some data
    functions (sortby, groupby, where, select, etc)
    """

    def __init__(self, *args, **kwargs):
        """A dictionary constructor and attributes declaration"""
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self  # give access to everything directly

    def __len__(self):
        """Returns the number of rows"""
        key0 = next(k for k in self.keys())
        return len(self[key0])  # type: ignore

    # set converters
    to_dict = convert.to_dict
    to_records = convert.convert_dict_to_structured_ndarray
    if convert.to_pandas:
        to_pandas = convert.to_pandas
    if convert.to_xarray:
        to_xarray = convert.to_xarray
    if convert.to_astropy_table:
        to_astropy_table = convert.to_astropy_table
    if convert.to_dask:
        to_dask = convert.to_dask
    if convert.to_vaex:
        to_vaex = convert.to_vaex
    if convert.pickle_dump:
        pickle_dump = convert.pickle_dump
    if convert.unpickle:
        pickle_load = classmethod(convert.unpickle)

    def _repr_html_(self) -> str | None:
        return self.to_pandas().head()._repr_html_()

    @property
    def nrows(self) -> int:
        """Number of rows in the dataset"""
        return len(self)

    @property
    def ncols(self) -> int:
        """Number of columns in the dataset"""
        return dict.__len__(self)

    @classmethod
    def from_lines(cls, iterable: Iterable[Dict]) -> "DictDataFrame":
        """Create a DataFrame object from its lines instead of columns

        Parameters
        ----------
        iterable: iterable
            sequence of lines with the same keys (expecting dict like
            structure)

        Returns
        -------
        df: DataFrame
            a new object
        """
        d = {}
        default_n = 0
        for line in iterable:
            for k in line.keys():
                d.setdefault(k, [np.atleast_1d(np.nan)] * default_n).append(
                    np.atleast_1d(line[k])
                )
        for k, v in dict.items(d):
            dict.__setitem__(d, k, np.squeeze(np.vstack(v)))

        return cls(d)

    def __repr__(self) -> str:
        """Return a string representation of the DataFrame."""
        txt = ["DataFrame ({0:s})\n".format(pretty_size_print(self.nbytes))]
        for k, v in self.items():
            try:
                txt.append(str((k, v.dtype, v.shape)))
            except AttributeError:
                txt.append(str((k, type(v))))
        return "\n".join(txt)

    @property
    def nbytes(self) -> int:
        """number of bytes of the object"""
        nbytes = sum(
            k.nbytes if hasattr(k, "nbytes") else sys.getsizeof(k)
            for k in self.__dict__.values()
        )
        return nbytes

    def __getitem__(self, k: str | Hashable):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            try:
                return self.evalexpr(k)
            except Exception:
                return self.__class__({a: v[k] for a, v in self.items()})

    @property
    def dtype(self) -> Dict[str | Hashable, np.dtype]:
        """the dtypes of each column of the dataset"""
        return dict((k, v.dtype) for (k, v) in self.items())

    @property
    def shape(self) -> Dict[str | Hashable, Tuple[int, ...]]:
        """dict of shapes"""
        return dict((k, v.shape) for (k, v) in self.items())

    def groupby(
        self, key: str | Hashable
    ) -> Iterator[Tuple[str | Hashable, "DictDataFrame"]]:
        """create an iterator which returns (key, DataFrame) grouped by each
        value of key(value)"""
        for k, index in self.arg_groupby(key):
            d = {a: b[index] for a, b in self.items()}
            yield k, self.__class__(d)

    def arg_groupby(
        self, key: str | Hashable
    ) -> Iterator[Tuple[str | Hashable, List[int]]]:
        """create an iterator which returns (key, index) grouped by each
        value of key(value)"""
        val = self.evalexpr(key)
        ind = sorted(zip(val, range(len(val))), key=lambda x: x[0])  # type: ignore

        for k, grp in itertools.groupby(ind, lambda x: x[0]):  # type: ignore
            index = [k[1] for k in grp]  # type: ignore
            yield k, index

    def __iter__(self) -> Iterator[Any]:
        """Iterator on the lines of the dataframe"""
        return self.iterlines()

    def iterlines(self) -> Iterator[Any]:
        """Iterator on the lines of the dataframe"""
        return self.lines

    @property
    def lines(self) -> Iterator[Any]:
        """Iterator on the lines of the dataframe"""
        for k in range(self.nrows):
            yield self[k]

    @property
    def rows(self) -> Iterator[Any]:
        """Iterator on the lines of the dataframe"""
        return self.lines

    @property
    def columns(self) -> ItemsView:
        """Iterator on the columns
        refers to :func:`dict.items`
        """
        return dict.items(self)

    def where(
        self, condition: str, condvars: Dict[str, Any] | None = None
    ) -> Iterator[Any]:
        """Read table data fulfilling the given `condition`.
        Only the rows fulfilling the `condition` are included in the result.

        Parameters
        ----------
        condition : str
            The evaluation of this condition should return True or False the
            condition can use field names and variables defined in condvars

        condvars: dict
            dictionary of variables necessary to evaluate the condition.

        Returns
        -------
        it: generator
            iterator on the query content. Each iteration contains one selected
            entry.

        ..note:
            there is no prior check on the variables and names
        """
        for line in self.lines:
            if eval(condition, dict(line), condvars):
                yield line

    def sortby(
        self, key: str | Hashable, reverse: bool = False, copy: bool = False
    ) -> Optional["DictDataFrame"]:
        """
        Parameters
        ----------
        key: str
            key to sort on. Must be in the data

        reverse: bool
            if set sort by decreasing order

        copy: bool
            if set returns a new dataframe

        Returns
        -------
        it: DataFrame or None
            new dataframe only if copy is False
        """
        val = self.evalexpr(key)
        ind = np.argsort(val)
        if reverse:
            ind = ind[::-1]
        if not copy:
            for k in self.keys():
                dict.__setitem__(self, k, dict.__getitem__(self, k)[ind])
        else:
            d = {}
            for k in self.keys():
                d[k] = dict.__getitem__(self, k)[ind]
            return self.__class__(d)

    def select(
        self, keys: str | Sequence[str], caseless: bool = False
    ) -> "DictDataFrame":
        """Read table data but returns only selected fields.

        Parameters
        ----------
        keys: str, sequence of str
            field names to select.
            Can be a single field names as follow:
            'RA', or ['RA', 'DEC'], or 'RA,DEC', or 'RA DEC'

        caseless: bool
            if set, do not pay attention to case.

        Returns
        -------
        df: DataFrame
            reduced dataframe

        ..note:
            there is no prior check on the variables and names
        """
        if keys == "*":
            return self

        if caseless:
            _keys = "".join([k.lower() for k in keys])
            df = self.__class__(
                dict((k, v) for k, v in self.items() if (k.lower() in _keys))
            )
        else:
            df = self.__class__(dict((k, v) for k, v in self.items() if k in keys))
        return df

    def multigroupby(
        self, key: Sequence[str | Hashable], *args
    ) -> Generator[Tuple[str | Hashable, "DictDataFrame"], None, None]:
        """
        Returns nested grouped DataFrames given the multiple keys

        Parameters
        ----------
        key1, key2, ...: sequence
            keys over which indexing the data

        Returns
        -------
        it: generator
            (key1, (key2, (... keyn, {})))
        """
        return _df_multigroupby(self, key, *args)  # type: ignore

    def aggregate(
        self,
        func: Callable,
        keys: Sequence[str | Hashable],
        args: Tuple[Any, ...] = (),
        kwargs: Dict[str, Any] = {},
    ):
        """Apply func on groups within the data

        Parameters
        ----------
        func: callable
            function to apply
        keys: sequence(str)
            sequence of keys defining the groups
        args: tuple
            optional arguments to func (will be given at the end)
        kwargs: dict
            optional keywords to func

        Returns
        -------
        seq: sequence
            flattened sequence of keys and value
            (key1, key2, ... keyn, {})
        """
        pv = [(k, list(v)) for k, v in self.multigroupby(*keys)]  # type: ignore
        return _df_multigroupby_aggregate(pv, func=func, args=args, kwargs=kwargs)

    @property
    def Plotter(self):
        """Plotter instance related to this dataset.
        Requires plotter add-on to work"""
        if Plotter is None:
            msg = "the add-on was not found, this property is not available"
            raise AttributeError(msg)
        else:
            return Plotter(self)

    def evalexpr(self, expr, exprvars=None, dtype=float):
        """evaluate expression based on the data and external variables
            all np function can be used (log, exp, pi...)

        Parameters
        ----------
        data: dict or dict-like structure
            data frame / dict-like structure containing named columns

        expr: str
            expression to evaluate on the table
            includes mathematical operations and attribute names

        exprvars: dictionary, optional
            A dictionary that replaces the local operands in current frame.

        dtype: dtype definition
            dtype of the output array

        Returns
        -------
        out : np.array
            array of the result
        """
        return evalexpr(self, expr, exprvars=exprvars, dtype=dtype)

    # for common interfaces
    evaluate = evalexpr

    def write(self, fname, keys=None, header=None, **kwargs):
        """Write DictDataFrame into file

        Parameters
        ---------
        fname: string
            file to write into

        keys: sequence, optional
            ordered sequence of columns

        header: string
            header to add to the file

        extension: str
            csv or txt/dat file
            auto guess from fname
            sets the delimiter and comments accordingly

        delimiter: str
            force  the column delimiter

        comments: str
            force the char defining comment line

        kwargs: dict
            forwarded to np.savetxt
        """
        extension = kwargs.pop("extension", None)
        if extension is None:
            extension = fname.split(".")[-1]
        if extension == "csv":
            comments = kwargs.pop("comments", "#")
            delimiter = kwargs.pop("delimiter", ",")
        elif extension in ("txt", "dat"):
            comments = kwargs.pop("comments", "#")
            delimiter = kwargs.pop("delimiter", " ")
        else:
            comments = ""
            delimiter = ""

        if keys is None:
            keys = self.keys()

        data = self.to_records(keys)
        hdr = f"{comments} {delimiter.join(data.dtype.names)}"  # type: ignore

        dtypes = (self.dtype[k] for k in keys)
        fmt = delimiter.join(["%" + k.kind.lower() for k in dtypes])

        # Monkey patch to help unicode/bytes/str mess
        fmt = fmt.replace("u", "s")

        np.savetxt(
            fname,
            data,
            delimiter=delimiter,
            header=(header or "") + hdr,
            fmt=fmt,
            comments="",
            **kwargs,
        )

    def join(self, key, other, key_other=None, columns_other=None, prefix=None):
        """
        Experimental joining of structures, (equivalent to SQL left outer join)
        Example:
        >>> x = np.arange(10)
        >>> y = x**2
        >>> z = x**3
        >>> df = DictDataFrame(x=x, y=y)
        >>> df2 = DictDataFrame(x=x[:4], z=z[:4])
        >>> df.join('x', df2, 'x', column_other=['z'])

        Parameters
        ----------
        key: str
            key for the left table (self)

        other: DictDataFrame
            Other dataset to join with (the right side of the outer join)

        key_other: str, optional
            key on which to join (default identical to key)

        columns_other: tuple, optional
            column names to add to the dataframe (default: all fields)
        prefix: str, optional
            add a prefix to the new column

        Returns
        -------
        self: DictDataFrame
            itself
        """
        N = len(self)
        N_other = len(other)
        if columns_other is None:
            columns_other = list(other.keys())
        if key_other is None:
            key_other = key

        # Bullet proofing existing data
        if prefix is None:
            for name in columns_other:
                if name in self:
                    msg = "Field {0:s} already exists.".format(name)
                    raise ValueError(msg)
            else:
                for name in columns_other:
                    new_name = "{0:s}{1:s}".format(prefix, name)
                    msg = "Field {0:s} already exists.".format(new_name)
                    if new_name in self:
                        raise ValueError(msg)

        # generate index
        key = self[key]
        key_other = other[key_other]
        index = dict(zip(key, range(N)))  # type: ignore
        index_other = dict(zip(key_other, range(N_other)))

        from_indices = np.zeros(N_other, dtype=np.int64)
        to_indices = np.zeros(N_other, dtype=np.int64)
        for i in range(N_other):
            if key_other[i] in index:
                to_indices[i] = index[key_other[i]]
                from_indices[i] = index_other[key_other[i]]
            else:
                to_indices[i] = -1
                from_indices[i] = -1

        mask = to_indices != -1
        to_indices = to_indices[mask]
        from_indices = from_indices[mask]

        dtypes = other.dtype
        for column_name in columns_other:
            dtype = dtypes[column_name]
            if np.issubdtype(dtype, np.inexact):
                data = np.zeros(N, dtype=dtype)
                data[:] = np.nan
                data[to_indices] = other[column_name][from_indices]
            else:
                data = np.ma.masked_all(N, dtype=dtype)
                values = other[column_name][from_indices]
                data[to_indices] = values
                data.mask[to_indices] = np.ma.masked
                if not np.ma.is_masked(data):
                    # forget the mask if we do not need it
                    data = data.data
            if prefix:
                new_name = "{0:s}{1:s}".format(prefix, column_name)
            else:
                new_name = column_name
            self[new_name] = data
        return self


def _df_multigroupby(
    ary: DictDataFrame, *args: Tuple[Any, ...]
) -> Generator[Tuple[Any, DictDataFrame], None, None] | Any:
    """
    Generate nested df based on multiple grouping keys

    Parameters
    ----------
    ary: dataFrame, dict like structure

    args: str or sequence
        column(s) to index the DF
    """
    if len(args) <= 0:
        yield ary
    else:
        if len(args) > 1:
            nested = True
        else:
            nested = False

        val = ary[args[0]]
        ind = sorted(zip(val, range(len(val))), key=lambda x: x[0])  # type: ignore

        for k, grp in itertools.groupby(ind, lambda x: x[0]):  # type: ignore
            index = [v[1] for v in grp]  # type: ignore
            d = ary.__class__(
                {a: np.array([b[i] for i in index]) for a, b in ary.items()}
            )
            if nested:
                yield k, _df_multigroupby(d, *args[1:])
            else:
                yield k, d


def _df_multigroupby_aggregate(
    pv: DictDataFrame | Dict[Any, Any] | List[Tuple[str | Hashable, Any]],
    func: Callable = lambda x: x,
    args=(),
    kwargs={},
):
    """
    Generate a flattened structure from multigroupby result

    Parameters
    ----------
    pv: dataFrame, dict like structure
    result from :func:`_df_multigroupby`

    func: callable
        reduce the data according to this function (default: identity)

    Returns
    -------
    seq: sequence
        flattened sequence of keys and value
    """

    def aggregate(a, b=()):
        data = []
        for k, v in a:
            if type(v) in (
                list,
                tuple,
            ):
                data.append(aggregate(v, b=(k,)))
            else:
                data.append(b + (k, func(v, *args, **kwargs)))
        return data

    return list(itertools.chain(*aggregate(pv)))


def evalexpr(
    data: DictDataFrame | Dict[Any, Any],
    expr: str,
    exprvars: Dict | None = None,
    dtype: npt.DTypeLike = float,
) -> npt.ArrayLike:
    """evaluate expression based on the data and external variables
        all np function can be used (log, exp, pi...)

    Parameters
    ----------
    data: dict or dict-like structure
        data frame / dict-like structure containing named columns

    expr: str
        expression to evaluate on the table
        includes mathematical operations and attribute names

    exprvars: dictionary, optional
        A dictionary that replaces the local operands in current frame.

    dtype: dtype definition
        dtype of the output array

    Returns
    -------
    out : np.array
        array of the result
    """
    _globals = {}
    keys = []
    if hasattr(data, "keys"):
        keys += list(data.keys())
    if hasattr(getattr(data, "dtype", None), "names"):
        keys += list(data.dtype.names)  # type: ignore
    if hasattr(data, "_aliases"):
        # SimpleTable specials
        keys += list(data._aliases.keys())  # type: ignore
    keys = set(keys)
    if expr in keys:
        return np.array(data[expr])
    for k in keys:
        if k in expr:
            _globals[k] = data[k]

    if exprvars is not None:
        if not hasattr(exprvars, "items"):
            msg = "Expecting a dict-like as condvars with an `items` method"
            raise AttributeError(msg)
        for k, v in exprvars.items():
            _globals[k] = v

    # evaluate expression, to obtain the final filter
    # r = np.empty( self.nrows, dtype=dtype)
    r = eval(expr, _globals, np.__dict__)

    return np.array(r, dtype=dtype)  # type: ignore
