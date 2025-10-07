from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from simpletable.dataframe.convert import ecsv


def make_basic_df():
    # Keep simple, well-supported dtypes for robust roundtrip
    return pd.DataFrame(
        {
            "i": np.arange(6, dtype=np.int64),
            "f": np.linspace(0.0, 1.0, 6),
            "s": np.array(list("abcdef"), dtype=object),  # object string
        }
    )


def test_generate_header_contains_meta_and_datatypes():
    df = make_basic_df()
    # Attach some meta to df.attrs
    df.attrs["NAME"] = "ECSVTest"
    df.attrs["author"] = "unit-test"

    hdr = ecsv.generate_header(df, extra="ok", version=1)
    # Header preamble and YAML structure
    assert hdr.startswith("# %ECSV 1.0")
    assert "datatype:" in hdr
    # Column names should be listed
    for col in df.columns:
        assert f"name: {col}" in hdr
    # Meta block should include attrs + additional kwargs
    assert "meta:" in hdr
    assert "NAME: ECSVTest" in hdr
    assert "author: unit-test" in hdr
    assert "extra: ok" in hdr


def test_write_then_read_roundtrip(tmp_path: Path):
    df = make_basic_df()
    # Add metadata into attrs
    df.attrs.update(
        {
            "NAME": "RoundtripTable",
            "project": "simpletable",
            "comment": "roundtrip test",
        }
    )

    out = tmp_path / "roundtrip.ecsv"
    # Write file with additional meta that should appear in header too
    ecsv.write(df, str(out), creator="test_ecsv")

    # Read full file back
    df2 = ecsv.read(str(out))
    assert isinstance(df2, pd.DataFrame)

    # Column equality and content
    assert list(df2.columns) == list(df.columns)
    # Use check_dtype=False to be robust to parser differences
    pdt.assert_series_equal(df2["i"], df["i"], check_dtype=False)
    pdt.assert_series_equal(df2["f"], df["f"], check_dtype=False)
    pdt.assert_series_equal(df2["s"], df["s"], check_dtype=False)

    # Meta should be restored in attrs
    meta = df2.attrs
    assert meta.get("NAME") == "RoundtripTable"
    assert meta.get("project") == "simpletable"
    assert meta.get("comment") == "roundtrip test"
    assert meta.get("creator") == "test_ecsv"


def test_read_chunks_preserves_meta(tmp_path: Path):
    df = make_basic_df()
    df.attrs.update({"NAME": "ChunkedTable", "info": "chunked-read"})

    out = tmp_path / "chunked.ecsv"
    ecsv.write(df, str(out), origin="unit")

    # Request chunked reading
    chunks = ecsv.read(str(out), chunksize=2)

    total = 0
    seen_meta = []
    for chunk in chunks:
        assert isinstance(chunk, pd.DataFrame)
        # Meta should be injected on each chunk
        attrs = chunk.attrs
        seen_meta.append(attrs)
        assert attrs.get("NAME") == "ChunkedTable"
        assert attrs.get("info") == "chunked-read"
        assert attrs.get("origin") == "unit"
        total += len(chunk)

    assert total == len(df)


def test_read_header_parsed_yaml(tmp_path: Path):
    df = make_basic_df()
    df.attrs.update({"NAME": "HeaderOnly", "tag": "parse"})

    out = tmp_path / "header.ecsv"
    ecsv.write(df, str(out), run="42")

    header = ecsv.read_header(str(out))
    # Expected top-level keys
    assert isinstance(header, dict)
    assert "delimiter" in header
    assert "datatype" in header
    assert "meta" in header

    # Datatype should have entries for each column
    names = [entry["name"] for entry in header["datatype"]]
    assert names == list(df.columns)

    # Meta contains attrs + extra
    meta = header["meta"]
    assert meta.get("NAME") == "HeaderOnly"
    assert meta.get("tag") == "parse"
    assert meta.get("run") == "42"


def test_dataframe_integration_write_read(tmp_path: Path):
    # Integration with DataFrame subclass users: write with ecsv using DataFrame
    # Then read back and build a pandas DataFrame; attrs should be present
    df = make_basic_df()
    df.attrs["NAME"] = "DFIntegration"

    out = tmp_path / "df_integration.ecsv"
    ecsv.write(df, str(out), purpose="integration")

    back = ecsv.read(str(out))
    pdt.assert_frame_equal(back, df, check_dtype=False)
    assert back.attrs.get("NAME") == "DFIntegration"  # pyright: ignore
    assert back.attrs.get("purpose") == "integration"  # pyright: ignore
