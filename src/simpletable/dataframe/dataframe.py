"""DataFrame class that is basically a wrapper around pandas.DataFrame which supports:
- Column aliases
- Units
- Case sensitivity
- search in the columns
- inputs and outputs into various additional formats: FITS, CSV/Ascii, ECSV
"""

import sys
import re
from typing import Any, Dict, Hashable, Iterable, List, Generator
import pandas as pd
import numpy.typing as npt
from io import TextIOWrapper

from pandas._typing import WriteBuffer

from .helpers import pretty_size_print
from .header import HeaderInfo
from . import convert


class DataFrame(pd.DataFrame):
    """DataFrame class that is basically a wrapper around pandas.DataFrame which supports:
    - Column aliases
    - Units
    - Case sensitivity
    - search in the columns
    - preserve attrs information at initialization from DataFrame
    - inputs and outputs into various additional formats: FITS, CSV, ECSV
    """

    def __init__(
        self,
        data: npt.NDArray | npt.ArrayLike | Iterable | Dict | pd.DataFrame,
        info: HeaderInfo | None = None,
        caseless: bool = False,
    ):
        super().__init__(data)
        attrs = getattr(data, "attrs", {})
        self.attrs.update(attrs)
        if info is not None:
            self.attrs.update(info.header)
            self.attrs["comments"] = info.comments
            self.attrs["aliases"] = info.alias
            self.attrs["units"] = info.units
        self.attrs["caseless"] = caseless
        self._clean_orphan_aliases()

    @property
    def nrows(self) -> int:
        return len(self)

    @property
    def ncols(self) -> int:
        return len(self.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nrows, self.ncols

    @property
    def colnames(self) -> list[str]:
        return list(self.columns)

    @property
    def nbytes(self):
        """number of bytes of the object"""
        n = sum(
            k.nbytes if hasattr(k, "nbytes") else sys.getsizeof(k)
            for k in self.__dict__.values()
        )
        n += self.memory_usage(True, True).sum()
        return n

    def resolve_alias(
        self, colname: str | bytes | Hashable | Any
    ) -> str | bytes | Hashable | Any:
        """Resolve an alias to the actual column name

        Parameters
        ----------
        colname: str | bytes | Hashable | Any
            Name or alias of the column

        Returns
        -------
        str | bytes | Hashable | Any
            Actual column name or initial value if not found
        """
        if not self.attrs.get("caseless", False):
            return self.attrs["aliases"].get(colname, colname)
        else:
            if isinstance(colname, (str, bytes)):
                for k, v in self.attrs["aliases"].items():
                    if v.lower() == colname.lower():
                        return k
        return colname

    def reverse_alias(
        self, colname: str | bytes | Hashable | Any
    ) -> str | bytes | Hashable | Any:
        """Returns aliases of a given column

        Parameters
        ----------
        colname: str | bytes | Hashable | Any
            Name or alias of the column

        Returns
        -------
        str | bytes | Hashable | Any
            Actual column name or initial value if not found
        """
        _colname = self.resolve_alias(colname)
        if _colname not in self.columns:
            raise KeyError(f"Column '{_colname}' not found")
        aliases = self.attrs.get("aliases", {})
        return tuple([k for (k, v) in aliases.items() if (v == _colname)])

    def set_column_comment(self, colname: str, comment: str):
        """Set the comment for a column referenced by name or alias

        Parameters
        ----------
        colname: str
            Name or alias of the column
        comment: str
            Comment to set for the column
        """
        self.attrs["comments"][self.resolve_alias(colname)] = comment

    def set_column_unit(self, colname: str, unit: str):
        """Set the unit for a column referenced by name or alias

        Parameters
        ----------
        colname: str
            Name or alias of the column
        unit: str
            Unit to set for the column
        """
        self.attrs["units"][self.resolve_alias(colname)] = str(unit)

    def __getitem__(self, key: str | Any) -> Any:
        """Getitem accounting for aliases"""
        return super().__getitem__(self.resolve_alias(key))

    def __setitem__(self, key: str | Any, value: Any) -> None:
        """Setitem accounting for aliases"""
        super().__setitem__(self.resolve_alias(key), value)

    def _clean_orphan_aliases(self):
        """Make sure remaining aliases are correctly links to some data"""
        aliases = self.attrs.get("aliases", {})
        names = self.dtypes.index
        self.attrs["aliases"] = {k: v for k, v in aliases.items() if v in names}

    def set_alias(self, alias: str, colname: str) -> None:
        """Set an alias for a column referenced by name or alias"""
        if colname not in self.keys():
            raise ValueError(f"Column '{colname}' does not exist")
        self.attrs["aliases"][alias] = colname

    def keys(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        regexp: str | Iterable[str] | None = None,
        full_match: bool = False,
        skip_aliases: bool = False,
    ) -> List[str] | pd.Index:
        """
        Return the data column names or a subset of it

        Parameters
        ----------
        regexp: str
            pattern to filter the keys with

        full_match: bool
            if set, use :func:`re.fullmatch` instead of :func:`re.findall`

        skip_aliases: bool
            include aliases if unset

        Try to apply the pattern at the start of the string, returning
        a match object, or None if no match was found.

        returns
        -------
        seq: sequence
            sequence of keys
        """
        if regexp in (None, "*"):
            return super().keys()
        if isinstance(regexp, (str, bytes)):
            lbls = list(self.columns)
            if not skip_aliases:
                lbls.extend(self.attrs.get("aliases", {}).keys())

            if regexp.count(",") > 0:
                re_ = regexp.split(",")
            elif regexp.strip().count(" ") > 0:
                re_ = regexp.split(" ")
            else:
                re_ = [regexp]

            fn = re.fullmatch if full_match else re.findall

            matches = []
            for subre in re_:
                matches.extend([lbl for lbl in lbls if fn(subre, lbl)])

            return matches
        elif hasattr(regexp, "__iter__"):
            matches = []
            for rk in regexp:
                matches.extend(
                    self.keys(rk, full_match=full_match, skip_aliases=skip_aliases)
                )
            return matches
        else:
            msg = "Unexpected type {0} for regexp".format(type(regexp))
            raise ValueError(msg)

    def pandas_info(
        self,
        verbose: bool | None = None,
        buf: WriteBuffer[str] | None = None,
        max_cols: int | None = None,
        memory_usage: bool | str | None = None,
        show_counts: bool | None = None,
    ):
        """Print a concise summary of a DataFrame.

        This method prints information about a DataFrame including
        the index dtype and columns, non-null values and memory usage.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the full summary. By default, the setting in
            ``pandas.options.display.max_info_columns`` is followed.
        buf : writable buffer, defaults to sys.stdout
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.
        max_cols : int, optional
            When to switch from the verbose to the truncated output. If the
            DataFrame has more than `max_cols` columns, the truncated output
            is used. By default, the setting in
            ``pandas.options.display.max_info_columns`` is used.
        memory_usage : bool, str, optional
            Specifies whether total memory usage of the DataFrame
            elements (including the index) should be displayed. By default,
            this follows the ``pandas.options.display.memory_usage`` setting.

            True always show memory usage. False never shows memory usage.
            A value of 'deep' is equivalent to "True with deep introspection".
            Memory usage is shown in human-readable units (base-2
            representation). Without deep introspection a memory estimation is
            made based in column dtype and number of rows assuming values
            consume the same memory amount for corresponding dtypes. With deep
            memory introspection, a real memory usage calculation is performed
            at the cost of computational resources. See the
            :ref:`Frequently Asked Questions <df-memory-usage>` for more
            details.
        show_counts : bool, optional
            Whether to show the non-null counts. By default, this is shown
            only if the DataFrame is smaller than
            ``pandas.options.display.max_info_rows`` and
            ``pandas.options.display.max_info_columns``. A value of True always
            shows the counts, and False never shows the counts.

        Returns
        -------
        None
            This method prints a summary of a DataFrame and returns None.

        See Also
        --------
        DataFrame.describe: Generate descriptive statistics of DataFrame
            columns.
        DataFrame.memory_usage: Memory usage of DataFrame columns.
        """
        return super().info(
            verbose=verbose,
            buf=buf,
            max_cols=max_cols,
            memory_usage=memory_usage,
            show_counts=show_counts,
        )

    def info(self, header: bool = False):  # pyright: ignore
        """prints information on the table"""
        txt = []
        name = self.attrs.get("NAME", "Noname")
        pprint_nbytes = pretty_size_print(self.nbytes)
        txt.append(f"""{name:s}""")
        txt.append(
            f"""       nrows={self.nrows:d}, ncols={self.ncols:d}, mem={pprint_nbytes:s}"""
        )
        # align keys properly
        if header:
            txt.append("Header:")
            ignore = ("aliases", "units")
            length = max([len(str(k)) for k in self.attrs.keys() if k not in ignore])
            fmt = "\t{{0:{0:d}s}} {{1}}".format(length)
            for k, v in self.attrs.items():
                if k not in ignore:
                    txt.append(fmt.format(k, str(v).strip()))

        # display column descriptions properly aligned
        units = self.attrs.get("units", {})
        desc = self.attrs.get("comments", {})
        values = [(k, units.get(k, ""), desc.get(k, "")) for k in self.colnames]
        lengths = [[len(val) for val in colval] for colval in values]
        lengths = list(map(max, (zip(*lengths))))
        txt.append("Columns (name, units, description):")
        fmt = "\t{{0:{0:d}s}} {{1:{1:d}s}} {{2:{2:d}s}}"
        fmt = fmt.format(*(k + 1 for k in lengths))
        for k, u, c in values:
            txt.append(fmt.format(k, u, c))

        aliases = self.attrs.get("aliases", {})
        if aliases:
            txt.append("Table contains alias(es):")
            for k, v in aliases.items():
                txt.append("\t{0:s} --> {1:s}".format(k, v))
        print("\n".join(txt))

    @classmethod
    def from_fits(cls, filename: str, extension_number: int = 1) -> "DataFrame":
        """Load a DataFrame from a FITS file.

        Parameters
        ----------
        filename : str
            The path to the FITS file.
        extension_number : int, optional
            The extension number to load, by default 1.

        Returns
        -------
        DataFrame
            The loaded DataFrame.
        """
        from_fits = getattr(convert.fits, "from_fits", None)
        if from_fits is None:
            raise ImportError("astropy (astropy.io.fits) is not installed")
        return DataFrame(*from_fits(filename, extension_number))

    if convert.fits.to_fits is not None:
        to_fits = convert.fits.to_fits

    to_ascii = convert.ascii.to_ascii
    to_csv = convert.ascii.to_csv  # pyright: ignore

    @classmethod
    def from_csv(
        cls, filepath_or_buffer: str, *, commented_header: bool = False, **kwargs
    ) -> "DataFrame":
        r"""
        Read a CSV file into a DataFrame while preserving header information

        Equivalent to `pd.read_csv` with preserved header information.

        Also supports optionally iterating or breaking of the file
        into chunks.

        Additional help can be found in the online docs for
        `IO Tools <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            Any valid string path is acceptable.
        commented_header: bool, default False
            Whether the column definition header line starts with a comment character.
        commentchar: str, default '#'
            Character to treat as a comment character.
        sep : str, default ','
            Character or regex pattern to treat as the delimiter. If ``sep=None``, the
            C engine cannot automatically detect the separator, but the Python
            parsing engine can, meaning the latter will be used and automatically
            detect the separator from only the first valid row of the file by
            Python's builtin sniffer tool, ``csv.Sniffer``.
            In addition, separators longer than 1 character and different from
            ``'\s+'`` will be interpreted as regular expressions and will also force
            the use of the Python parsing engine. Note that regex delimiters are prone
            to ignoring quoted data. Regex example: ``'\r\t'``.

        Returns
        -------
        DataFrame: pd.DataFrame
            The parsed data as a pd.DataFrame.
        header : HeaderInfo
            The header information extracted from the file.

        See also
        --------
        from_ascii: Read a ASCII file into a DataFrame
        """
        return cls(
            *convert.ascii.from_csv(
                filepath_or_buffer, commented_header=commented_header, **kwargs
            )
        )

    @classmethod
    def from_ascii(
        cls, filepath_or_buffer: str, *, commented_header: bool = False, **kwargs
    ) -> "DataFrame":
        """Read an ASCII file into a DataFrame.

        from_csv with delimiter set to " " by default

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            Any valid string path is acceptable. The string could be a URL. Valid
            URL schemes include http, ftp, s3, and file. For file URLs, a host is
            expected. A local file could be:
            ``file://localhost/path/to/table.csv``.
        commented_header : bool, default False
            Whether the header is commented or not.
        **kwargs : dict
            Additional keyword arguments passed to ``pd.read_csv``.

        Returns
        -------
        DataFrame: pd.DataFrame
            The parsed data as a pd.DataFrame.
        header : HeaderInfo
            The header information extracted from the file.

        See also
        --------
        from_csv: Read a CSV file into a DataFrame
        """
        return cls(
            *convert.ascii.from_ascii(
                filepath_or_buffer, commented_header=commented_header, **kwargs
            )
        )

    @classmethod
    def from_ecsv(
        cls, fname: str, **kwargs
    ) -> "DataFrame" | Generator["DataFrame", Any, None]:
        """Read the content of an Enhanced Character Separated Values

        Parameters
        ----------
        fname:  str
            The name of the file to read.

        Returns
        -------
        data:  pd.DataFrame | Generator[pd.DataFrame, Any, None]
            The data read from the file.
            The chunked data read from the file.
        """
        return cls(convert.ecsv.read(fname, **kwargs))

    def to_ecsv(
        self,
        df: pd.DataFrame,
        fname: str | TextIOWrapper,
        mode: str = "w",
        **meta,
    ):
        """output data into ecsv file

        Parameters
        ----------
        df: pd.DataFrame
            data to be written to the file.
        fname: str
            the name of the file to write.
        mode: str
            the mode to open the file.
        meta: dict
            meta data to be written to the header.
        """
        convert.ecsv.write(df, fname, mode=mode, **meta)


def testing():
    filename = "/Users/fouesneau/Downloads/atlas9-munari.hires.grid.fits"
    return DataFrame.from_fits(filename)
