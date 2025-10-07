import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

# Skip this test module if astropy is not installed
pytest.importorskip("astropy")

from simpletable import DataFrame
from simpletable.dataframe.convert import fits as fits_conv
from simpletable.dataframe.header import HeaderInfo


def make_df_with_attrs():
    df = pd.DataFrame(
        {
            "alpha": np.arange(5, dtype=np.int64),
            "beta": np.linspace(0.0, 1.0, 5),
        }
    )
    df.attrs.update(
        {
            "NAME": "FitsTable",
            "aliases": {"al": "alpha", "be": "beta"},
            "units": {"alpha": "ct", "beta": "s"},
            "comments": {"alpha": "first column", "beta": "second column"},
        }
    )
    return df


def make_st_df():
    df = make_df_with_attrs()
    info = HeaderInfo(
        header={
            k: v
            for k, v in df.attrs.items()
            if k not in ("aliases", "units", "comments")
        },
        alias=df.attrs["aliases"],
        units=df.attrs["units"],
        comments=df.attrs["comments"],
    )
    return DataFrame(df, info=info)


def test_fits_generate_header_contains_units_alias_and_extname():
    df = make_df_with_attrs()

    hdr = fits_conv.fits_generate_header(df)

    # EXTNAME should reflect NAME
    assert hdr.get("EXTNAME") == "FitsTable"

    # Column name and unit cards
    assert hdr.get("TTYPE1") == "alpha"
    assert hdr.get("TTYPE2") == "beta"
    assert hdr.get("TUNIT1") == "ct"
    assert hdr.get("TUNIT2") == "s"

    # Comments attached to TTYPE cards should reflect column comments
    # via the Header comments accessor
    assert hdr.comments["TTYPE1"] == "first column"
    assert hdr.comments["TTYPE2"] == "second column"

    # Aliases encoded as ALIASn cards
    alias_values = [
        card.value for card in hdr.cards if card.keyword.startswith("ALIAS")
    ]
    assert "al=alpha" in alias_values
    assert "be=beta" in alias_values


def test_to_fits_and_from_fits_roundtrip_dataframe(tmp_path):
    # Build a simpletable.DataFrame with metadata
    df = make_st_df()

    # Write to FITS
    out = tmp_path / "roundtrip.fits"
    # Use the DataFrame class facade which forwards to converter
    DataFrame.to_fits(df, str(out), overwrite=True, index=False)

    # Read back through the DataFrame API
    df2 = DataFrame.from_fits(str(out))
    df2.info()

    # Columns and data equality (be lenient on dtype)
    assert list(df2.columns) == list(df.columns)
    pdt.assert_series_equal(df2["alpha"], df["alpha"], check_dtype=False)
    pdt.assert_series_equal(df2["beta"], df["beta"], check_dtype=False)

    # Metadata restored in attrs
    assert df2.attrs.get("NAME") == df.attrs["NAME"].upper()
    assert df2.attrs.get("units", {}).get("alpha") == "ct"
    assert df2.attrs.get("units", {}).get("beta") == "s"
    assert df2.attrs.get("comments", {}).get("alpha") == "first column"
    assert df2.attrs.get("comments", {}).get("beta") == "second column"
    assert df2.attrs.get("aliases", {}).get("al") == "alpha"
    assert df2.attrs.get("aliases", {}).get("be") == "beta"

    # Aliases work for __getitem__
    np.testing.assert_array_equal(df2["al"].to_numpy(), df2["alpha"].to_numpy())
    np.testing.assert_array_equal(df2["be"].to_numpy(), df2["beta"].to_numpy())
