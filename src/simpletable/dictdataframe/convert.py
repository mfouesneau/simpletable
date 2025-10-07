"""Module for converting dictdataframes to other formats."""

import numpy as np


def to_dict(self, keys=None, contiguous=False):
    """Construct a dictionary from this dataframe with contiguous arrays

    Parameters
    ----------
    keys: sequence, optional
        ordered subset of columns to export

    contiguous: boolean
        make sure each value is a contiguous numpy array object
        (C-aligned)

    Returns
    -------
    data: dict
        converted data
    """
    if keys is None:
        keys = self.keys()
    if contiguous:
        return {k: np.ascontiguousarray(self[k]) for k in keys}
    return {k: self[k] for k in keys}


def convert_dict_to_structured_ndarray(data, keys=None):
    """convert_dict_to_structured_ndarray

    Parameters
    ----------
    data: dictionary like object
        data structure which provides iteritems and itervalues
    keys: sequence, optional
        ordered subset of columns to export

    returns
    -------
    tab: structured ndarray
        structured numpy array
    """
    newdtype = []
    if keys is None:
        keys = data.keys()
    for key in keys:
        _dk = data[key]
        dtype = data.dtype[key]
        # unknown type is converted to text
        if dtype.type == np.object_:
            if len(data) == 0:
                longest = 0
            else:
                longest = len(max(_dk, key=len))
                _dk = _dk.astype("|%iS" % longest)
        if _dk.ndim > 1:
            newdtype.append((str(key), _dk.dtype, (_dk.shape[1],)))
        else:
            newdtype.append((str(key), _dk.dtype))
    tab = np.rec.fromarrays((data[k] for k in keys), dtype=newdtype)
    return tab


try:
    from pandas import DataFrame  # pyright: ignore[reportMissingImports]

    def to_pandas(self, **kwargs):
        """Construct a pandas dataframe

        Parameters
        ----------
        data : ndarray
            (structured dtype), list of tuples, dict, or DataFrame
        keys: sequence, optional
            ordered subset of columns to export
        index : string, list of fields, array-like
            Field of array to use as the index, alternately a specific set of
            input labels to use
        exclude : sequence, default None
            Columns or fields to exclude
        columns : sequence, default None
            Column names to use. If the passed data do not have names
            associated with them, this argument provides names for the
            columns. Otherwise this argument indicates the order of the columns
            in the result (any names not found in the data will become all-NA
            columns)
        coerce_float : boolean, default False
            Attempt to convert values to non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets

        Returns
        -------
        df : DataFrame
        """
        keys = kwargs.pop("keys", None)
        return DataFrame.from_dict(self.to_dict(keys=keys), **kwargs)

except ImportError:
    to_pandas = None  # pyright: ignore[reportAssignmentType]

try:
    from xarray import Dataset  # pyright: ignore[reportMissingImports]

    def to_xarray(self, **kwargs):
        """Construct an xarray dataset

        Each column will be converted into an independent variable in the
        Dataset. If the dataframe's index is a MultiIndex, it will be expanded
        into a tensor product of one-dimensional indices (filling in missing
        values with NaN). This method will produce a Dataset very similar to
        that on which the 'to_dataframe' method was called, except with
        possibly redundant dimensions (since all dataset variables will have
        the same dimensionality).

        Parameters
        ----------
        data : ndarray
            (structured dtype), list of tuples, dict, or DataFrame
        keys: sequence, optional
            ordered subset of columns to export
        index : string, list of fields, array-like
            Field of array to use as the index, alternately a specific set of
            input labels to use
        exclude : sequence, default None
            Columns or fields to exclude
        columns : sequence, default None
            Column names to use. If the passed data do not have names
            associated with them, this argument provides names for the
            columns. Otherwise this argument indicates the order of the columns
            in the result (any names not found in the data will become all-NA
            columns)
        coerce_float : boolean, default False
            Attempt to convert values to non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets

        Returns
        -------
        df : DataFrame
        """
        return Dataset.from_dataframe(self.to_pandas(**kwargs))
except ImportError:
    to_xarray = None  # pyright: ignore[reportAssignmentType]

try:
    import vaex  # pyright: ignore[reportMissingImports]

    def to_vaex(self, **kwargs):
        """
        Create an in memory Vaex dataset

        Parameters
        ----------
        name: str
            unique for the dataset
        keys: sequence, optional
            ordered subset of columns to export

        Returns
        -------
        df: vaex.DataSetArrays
            vaex dataset
        """
        return vaex.from_arrays(**self.to_dict(contiguous=True, **kwargs))
except ImportError:
    to_vaex = None  # pyright: ignore[reportAssignmentType]

try:
    from dask import dataframe  # pyright: ignore[reportMissingImports]

    def to_dask(self, **kwargs):
        """Construct a Dask DataFrame

        This splits an in-memory Pandas dataframe into several parts and
        constructs a dask.dataframe from those parts on which Dask.dataframe
        can operate in parallel.

        Note that, despite parallelism, Dask.dataframe may not always be faster
        than Pandas.  We recommend that you stay with Pandas for as long as
        possible before switching to Dask.dataframe.

        Parameters
        ----------
        keys: sequence, optional
            ordered subset of columns to export
        npartitions : int, optional
            The number of partitions of the index to create. Note that
            depending on the size and index of the dataframe, the output may
            have fewer partitions than requested.
        chunksize : int, optional
            The size of the partitions of the index.
        sort: bool
            Sort input first to obtain cleanly divided partitions or don't sort
            and don't get cleanly divided partitions
        name: string, optional
            An optional keyname for the dataframe.  Defaults to hashing the
            input

        Returns
        -------
        dask.DataFrame or dask.Series
            A dask DataFrame/Series partitioned along the index
        """
        keys = kwargs.pop("keys", None)
        return dataframe.from_pandas(self.to_pandas(keys=keys), **kwargs)
except ImportError:
    to_dask = None  # pyright: ignore[reportAssignmentType]

try:
    from astropy.table import Table  # pyright: ignore[reportMissingImports]

    def to_astropy_table(self, **kwargs):
        """
        A class to represent tables of heterogeneous data.

        `astropy.table.Table` provides a class for heterogeneous tabular data,
        making use of a `numpy` structured array internally to store the data
        values.  A key enhancement provided by the `Table` class is the ability
        to easily modify the structure of the table by adding or removing
        columns, or adding new rows of data.  In addition table and column
        metadata are fully supported.

        Parameters
        ----------
        masked : bool, optional
            Specify whether the table is masked.
        names : list, optional
            Specify column names
        dtype : list, optional
            Specify column data types
        meta : dict, optional
            Metadata associated with the table.
        copy : bool, optional
            Copy the input data (default=True).
        rows : numpy ndarray, list of lists, optional
            Row-oriented data for table instead of ``data`` argument
        copy_indices : bool, optional
            Copy any indices in the input data (default=True)
        **kwargs : dict, optional
            Additional keyword args when converting table-like object

        Returns
        -------
        df: astropy.table.Table
            dataframe
        """
        keys = kwargs.pop("keys", None)
        return Table(self.to_records(keys=keys), **kwargs)
except ImportError:
    to_astropy_table = None  # pyright: ignore[reportAssignmentType]

try:
    import pickle

    def pickle_dump(self, fname):
        """create a pickle dump of the dataset"""
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def unpickle(cls, fname):
        """restore a previously pickled object"""
        with open(fname, "rb") as f:
            return pickle.load(f)
except ImportError:
    unpickle = None  # pyright: ignore[reportAssignmentType]
    pickle_dump = None  # pyright: ignore[reportAssignmentType]
