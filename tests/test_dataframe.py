import io
import re

import numpy as np
import pandas as pd
import pytest

from simpletable import DataFrame
from simpletable.dataframe.header import HeaderInfo


def make_df_with_meta():
    # Build a simple DataFrame with metadata containers present
    pdf = pd.DataFrame(
        {
            "alpha": np.arange(5, dtype=np.int64),
            "beta": np.linspace(0.0, 1.0, 5),
            "gamma": list("abcde"),
        }
    )
    info = HeaderInfo(
        header={"NAME": "MyTable", "COMMENT": "line1\nline2"},
        alias={"a": "alpha", "b": "beta"},
        units={"alpha": "ct", "beta": "s", "gamma": ""},
        comments={"alpha": "first", "beta": "second"},
    )
    return DataFrame(pdf, info=info)


def test_basic_properties_and_aliases():
    df = make_df_with_meta()

    # nrows/ncols/shape/colnames
    assert df.nrows == 5
    assert df.ncols == 3
    assert df.shape == (5, 3)
    assert df.colnames == ["alpha", "beta", "gamma"]

    # nbytes is an int and at least dataframe memory usage
    assert isinstance(df.nbytes, (int, np.integer))
    assert df.nbytes >= df.memory_usage(index=True, deep=True).sum()

    # alias resolution in __getitem__ and __setitem__
    np.testing.assert_array_equal(df["a"].to_numpy(), df["alpha"].to_numpy())
    np.testing.assert_array_equal(df["b"].to_numpy(), df["beta"].to_numpy())

    # set_alias then get via alias
    df.set_alias("g", "gamma")
    assert "gamma" in df.columns
    np.testing.assert_array_equal(df["g"].to_numpy(), df["gamma"].to_numpy())

    # reverse_alias lists all aliases for a col
    df.set_alias("ra1", "alpha")
    df.set_alias("ra2", "alpha")
    aliases = set(df.reverse_alias("alpha"))  # pyright: ignore[reportArgumentType]
    assert {"a", "ra1", "ra2"}.issubset(aliases)

    # reverse_alias raises if col does not exist
    with pytest.raises(KeyError):
        df.reverse_alias("does_not_exist")

    # orphan alias cleanup
    df.attrs["aliases"]["dangling"] = "nope"
    df._clean_orphan_aliases()
    assert "dangling" not in df.attrs["aliases"]


def test_keys_filtering_and_alias_inclusion():
    df = make_df_with_meta()

    # default returns pandas Index
    keys_default = df.keys()
    assert isinstance(keys_default, pd.Index)
    assert list(keys_default) == ["alpha", "beta", "gamma"]

    # regex startswith using re.match semantics
    ks = df.keys("^a")
    assert ks == ["alpha", "a"] or ks == ["a", "alpha"]  # alias may interleave

    # multiple patterns separated by comma
    ks_multi = df.keys("^a,^g")
    assert set(ks_multi) >= {"alpha", "gamma", "a"}

    # skip aliases
    ks_no_alias = df.keys("^a", skip_aliases=True)
    assert ks_no_alias == ["alpha"]

    # list/iterable of patterns merges results
    ks_iter = df.keys(["^a", "ma$"], skip_aliases=True)
    assert set(ks_iter) == {"alpha", "gamma"}

    # full_match
    ks_full = df.keys("alpha", full_match=True)
    assert ks_full == ["alpha"]


def test_pandas_info_and_custom_info_output(capsys):
    df = make_df_with_meta()

    # pandas_info delegates to pandas.DataFrame.info
    buf = io.StringIO()
    df.pandas_info(buf=buf)
    out = buf.getvalue()
    assert "RangeIndex" in out
    assert "alpha" in out
    assert "beta" in out

    # custom info prints header and columns
    df.info(header=True)
    printed = capsys.readouterr().out
    assert "MyTable" in printed
    assert re.search(r"nrows=\d+, ncols=\d+", printed) is not None
    assert "Columns (name, units, description):" in printed
    assert "alpha" in printed and "ct" in printed and "first" in printed
    assert "Table contains alias(es):" in printed
    # pretty size is human readable like "KB", "MB", etc. or Bytes
    assert re.search(r"mem=\d+(\.\d+)?\s*(Bytes|KB|MB|GB|TB|PB|EB|ZB|YB)", printed)


@pytest.mark.parametrize("index_flag", [False])  # ensure no index gets serialized
def test_csv_roundtrip_with_header_attrs(index_flag):
    df = make_df_with_meta()

    # Serialize to a CSV-like text buffer (comma sep). Header is not commented for the column line.
    buf = io.StringIO()
    df.to_csv(buf, index=index_flag)
    content = buf.getvalue()
    assert "%ECSV" not in content  # ensure not ECSV
    assert "# alias" in content or "alias" in content  # alias section present
    assert "alpha" in content and "beta" in content

    # Rewind for reading
    buf.seek(0)

    # Read back while indicating column header is not commented
    df2 = DataFrame.from_csv(buf, commented_header=False)  # pyright: ignore[reportArgumentType]
    assert isinstance(df2, DataFrame)
    assert list(df2.columns) == list(df.columns)

    # Values round-trip (types may differ slightly but numerics should be equal)
    pd.testing.assert_series_equal(df2["alpha"], df["alpha"], check_dtype=False)
    pd.testing.assert_series_equal(df2["beta"], df["beta"], check_dtype=False)
    pd.testing.assert_series_equal(df2["gamma"], df["gamma"], check_dtype=False)

    # Metadata round-trip
    assert df2.attrs.get("NAME") == "MyTable"
    assert df2.attrs.get("units", {}).get("alpha") == "ct"
    assert df2.attrs.get("comments", {}).get("beta") == "second"
    assert df2.attrs.get("aliases", {}).get("a") == "alpha"
    # Alias should work on read-back
    np.testing.assert_array_equal(df2["a"].to_numpy(), df2["alpha"].to_numpy())


@pytest.mark.parametrize("index_flag", [False])  # ensure no index gets serialized
def test_ascii_roundtrip_with_header_attrs(index_flag):
    df = make_df_with_meta()

    # Serialize to a space-separated ASCII buffer. Column header is commented.
    buf = io.StringIO()
    df.to_ascii(buf, index=index_flag)
    content = buf.getvalue()
    assert content.startswith("#")
    assert "# alias" in content
    assert "# alpha" in content  # column meta lines
    assert re.search(r"^#\s+alpha\s+beta\s+gamma\s*$", content, re.MULTILINE)

    # Rewind for reading
    buf.seek(0)

    # Read back indicating header line is commented
    df2 = DataFrame.from_ascii(buf, commented_header=True)  # pyright: ignore[reportArgumentType]
    assert isinstance(df2, DataFrame)
    assert list(df2.columns) == list(df.columns)

    # Values round-trip
    pd.testing.assert_series_equal(df2["alpha"], df["alpha"], check_dtype=False)
    pd.testing.assert_series_equal(df2["beta"], df["beta"], check_dtype=False)
    pd.testing.assert_series_equal(df2["gamma"], df["gamma"], check_dtype=False)

    # Metadata round-trip
    assert df2.attrs.get("NAME") == "MyTable"
    assert df2.attrs.get("units", {}).get("alpha") == "ct"
    assert df2.attrs.get("comments", {}).get("beta") == "second"
    assert df2.attrs.get("aliases", {}).get("a") == "alpha"


def test_set_column_unit_and_comment():
    df = make_df_with_meta()
    df.set_column_unit("alpha", "m")
    df.set_column_comment("beta", "time column")
    assert df.attrs["units"]["alpha"] == "m"
    assert df.attrs["comments"]["beta"] == "time column"
