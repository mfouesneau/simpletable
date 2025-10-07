from simpletable.dataframe.convert import fits
import importlib
import pandas as pd
import numpy as np

importlib.reload(fits)


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


df = make_df_with_attrs()
hdu = fits.fits_generate_hdu(df, index=False)

table_name = df.attrs.get("NAME", None)
records = df.to_records(index=False)
header = fits.fits_generate_header(df)
