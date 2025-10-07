import io
import re

import numpy as np
import pandas as pd
import pandas.testing as pdt

from pandas import DataFrame as pd_DataFrame
from simpletable.dataframe.convert import ascii
from simpletable.dataframe.header import HeaderInfo
from simpletable import DataFrame


def make_df_with_attrs():
    df = pd_DataFrame(
        {
            "alpha": np.arange(4, dtype=np.int64),
            "beta": np.linspace(0.0, 1.0, 4),
            "gamma": np.array(list("wxyz"), dtype=object),
        }
    )
    # Attach metadata that ascii_generate_header will serialize
    df.attrs.update(
        {
            "NAME": "AsciiTable",
            "PROJECT": "simpletable",
            "HISTORY": "created\nupdated",
            "COMMENT": "example\nunit-test",
            "aliases": {"a": "alpha", "b": "beta"},
            "units": {"alpha": "ct", "beta": "s"},
            "comments": {"alpha": "count", "beta": "time"},
        }
    )
    return df


def test_ascii_generate_header_and_parse_roundtrip_commented_header():
    df = make_df_with_attrs()

    # Generate header with commented column line and space delimiter
    hdr = ascii.ascii_generate_header(
        df, comments="#", delimiter=" ", commented_header=True
    )
    # Compose a complete file content: header + data rows (no header repeated)
    buf = io.StringIO()  # note io.StringIO(txt) does not set the cursor position
    buf.write(hdr)
    df.to_csv(buf, index=False, sep=" ", header=False, mode="a")

    # reset cursor position to start of buffer
    buf.seek(0)

    # Parse header from a stream and ensure it is correct
    nlines, info, names = ascii.ascii_read_header(
        buf, commentchar="#", delimiter=" ", commented_header=True
    )

    # For streams, function rewinds to start of data and sets skiprows to 0
    assert nlines == 0
    assert names == list(df.columns)

    # HeaderInfo content
    assert isinstance(info, HeaderInfo)
    # Header keys are upper-cased by generator
    assert info.header.get("NAME") == "AsciiTable"
    assert info.header.get("PROJECT") == "simpletable"
    # COMMENT and HISTORY aggregated across lines
    assert "example" in info.header.get("COMMENT", "")
    assert "unit-test" in info.header.get("COMMENT", "")
    assert "created" in info.header.get("HISTORY", "")
    assert "updated" in info.header.get("HISTORY", "")

    # Column metadata lines (## ...)
    assert info.units.get("alpha") == "ct"
    assert info.units.get("beta") == "s"
    assert info.comments.get("alpha") == "count"
    assert info.comments.get("beta") == "time"

    # Aliases
    assert info.alias.get("a") == "alpha"
    assert info.alias.get("b") == "beta"

    # After parsing header, ensure the remaining buffer reads the data correctly
    df_read = pd.read_csv(buf, delimiter=" ", header=None, names=names)
    pdt.assert_frame_equal(df_read, df, check_dtype=False)


def test_ascii_generate_header_and_parse_roundtrip_uncommented_colnames():
    pdf = make_df_with_attrs()

    # Generate header with non-commented column line (CSV-like)
    hdr = ascii.ascii_generate_header(
        pdf, comments="#", delimiter=",", commented_header=False
    )
    data_part = pdf.to_csv(index=False, sep=",", header=False)
    content = hdr + data_part

    buf = io.StringIO(content)
    nlines, info, names = ascii.ascii_read_header(
        buf, commentchar="#", delimiter=",", commented_header=False
    )
    # For streams, skiprows=0 and names is the first real data line (colnames)
    assert nlines == 0
    assert names == list(pdf.columns)

    # Ensure metadata parsed properly
    assert info.header.get("NAME") == "AsciiTable"
    assert info.alias.get("a") == "alpha"
    assert info.units.get("beta") == "s"

    # Now parse the remaining data with pandas
    df_read = pd.read_csv(buf, delimiter=",", header=None, names=names)
    pdt.assert_frame_equal(df_read, pdf, check_dtype=False)


def test_dataframe_facade_ascii_roundtrip_preserves_meta():
    # Build a DataFrame subclass instance with metadata
    pdf = make_df_with_attrs()
    info = HeaderInfo(
        header={
            k: v
            for k, v in pdf.attrs.items()
            if k not in ("aliases", "units", "comments")
        },
        alias=pdf.attrs["aliases"],
        units=pdf.attrs["units"],
        comments=pdf.attrs["comments"],
    )
    df = DataFrame(pdf, info=info)

    # Write to ASCII buffer (space-separated) via facade; header is commented
    buf = io.StringIO()
    df.to_ascii(buf, index=False)
    text = buf.getvalue()
    print(text)

    # Header and alias lines present
    assert text.startswith("#")
    assert "# alias" in text
    # Column meta lines (double comment char '##')
    assert re.search(r"^##\s*alpha\s*\tct\tcount\s*$", text, re.MULTILINE)

    # Read back through facade with commented header
    buf.seek(0)
    df2 = DataFrame.from_ascii(buf, commented_header=True)  # pyright: ignore

    assert isinstance(df2, DataFrame)
    assert list(df2.columns) == list(df.columns)

    # Content equality (robust to dtype differences)
    pdt.assert_series_equal(df2["alpha"], df["alpha"], check_dtype=False)
    pdt.assert_series_equal(df2["beta"], df["beta"], check_dtype=False)
    pdt.assert_series_equal(df2["gamma"], df["gamma"], check_dtype=False)

    # Metadata round-trip through attrs
    assert df2.attrs.get("NAME") == "AsciiTable"
    assert df2.attrs.get("PROJECT") == "simpletable"
    # Column metadata
    units = df2.attrs.get("units", {})
    comments = df2.attrs.get("comments", {})
    assert units.get("alpha") == "ct"
    assert comments.get("beta") == "time"
    # Aliases usable through DataFrame API
    np.testing.assert_array_equal(df2["a"].to_numpy(), df2["alpha"].to_numpy())


def test_ascii_module_from_ascii_and_from_csv_paths_streams():
    # Validate that ascii.from_ascii and ascii.from_csv can read from StringIO streams
    df = make_df_with_attrs()
    # Create CSV-like header (uncommented column names) and data content
    hdr_csv = ascii.ascii_generate_header(
        df, comments="#", delimiter=",", commented_header=False
    )
    buf_csv = io.StringIO()  # note io.StringIO(txt) does not set the cursor position
    buf_csv.write(hdr_csv)
    df.to_csv(buf_csv, index=False, header=False)

    # reset cursor position to start of buffer
    buf_csv.seek(0)

    # from_csv should parse names and header info from the stream
    df_csv, hdr_info_csv = ascii.from_csv(buf_csv, commented_header=False)  # pyright: ignore
    assert list(df_csv.columns) == list(df.columns)
    pdt.assert_frame_equal(df_csv, df, check_dtype=False)
    assert hdr_info_csv.header.get("NAME") == "AsciiTable"
    assert hdr_info_csv.alias.get("a") == "alpha"

    # Create ASCII-like header (commented column names) and data content
    hdr_ascii = ascii.ascii_generate_header(
        df, comments="#", delimiter=" ", commented_header=True
    )
    content_ascii = hdr_ascii + df.to_csv(index=False, sep=" ", header=False)
    buf_ascii = io.StringIO(content_ascii)

    # from_ascii should parse correctly as well
    df_ascii, hdr_info_ascii = ascii.from_ascii(buf_ascii, commented_header=True)  # pyright: ignore
    assert list(df_ascii.columns) == list(df.columns)
    pdt.assert_frame_equal(df_ascii, df, check_dtype=False)
    assert hdr_info_ascii.units.get("beta") == "s"
    assert hdr_info_ascii.comments.get("alpha") == "count"
