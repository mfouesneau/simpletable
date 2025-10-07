import numpy as np

from simpletable import DictDataFrame


def make_df(n=6):
    # Small, heterogeneous dataset
    return DictDataFrame(
        x=np.arange(n),
        y=np.arange(n) ** 2,
        z=np.arange(n).astype(float) / 2.0,
        s=np.array(list("abcdef")[:n]),
        grp=np.array(list("aaabbb")[:n]),
    )


def test_construction_and_basic_properties():
    df = make_df(5)

    # nrows/ncols/len
    assert df.nrows == 5
    assert df.ncols == 5
    assert len(df) == 5

    # dtype and shape maps
    dtypes = df.dtype
    assert set(dtypes.keys()) == {"x", "y", "z", "s", "grp"}
    assert dtypes["x"].kind in ("i", "l")  # integer type
    assert dtypes["z"].kind == "f"
    assert dtypes["s"].kind in ("U", "S")  # string-like
    assert isinstance(df.shape, dict)
    for k, shp in df.shape.items():
        assert shp[0] == df.nrows, f"Column {k} has inconsistent first dimension."

    # __getitem__ for a column and a row
    np.testing.assert_array_equal(df["x"], np.arange(5))
    row0 = df[0]
    assert isinstance(row0, DictDataFrame)
    assert row0["x"] == 0
    assert row0["s"] == "a"

    # nbytes returns an int and at least accounts for arrays
    assert isinstance(df.nbytes, int)
    assert df.nbytes > 0


def test_from_lines_roundtrip():
    lines = [
        {"a": 1, "b": 10.0},
        {"a": 2, "b": 20.0},
        {"a": 3, "b": 30.0},
    ]
    df = DictDataFrame.from_lines(lines)
    np.testing.assert_array_equal(df["a"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(df["b"], np.array([10.0, 20.0, 30.0]))


def test_select_variants_and_caseless():
    df = make_df(4)

    # Select by list
    sub = df.select(["x", "s"])
    assert set(sub.keys()) == {"x", "s"}

    # Select by comma-separated string
    sub2 = df.select("x,s")
    assert set(sub2.keys()) == {"x", "s"}

    # Select all
    assert df.select("*") is df

    # Caseless selection
    sub3 = df.select("X,S", caseless=True)
    assert set(sub3.keys()) == {"x", "s"}


def test_groupby_counts_and_values():
    df = make_df(6)  # grp: a,a,a,b,b,b
    groups = list(df.groupby("grp"))
    keys = [k for k, _ in groups]
    assert keys == ["a", "b"]
    sizes = [g.nrows for _, g in groups]
    assert sizes == [3, 3]
    # Validate content mapping for a single group
    g_a = dict(groups)["a"]
    np.testing.assert_array_equal(g_a["grp"], np.array(list("aaa")))
    np.testing.assert_array_equal(g_a["x"], np.array([0, 1, 2]))


def test_where_conditions_and_condvars():
    df = make_df(6)

    # Even x only
    evens = list(df.where("x % 2 == 0"))
    assert len(evens) == 3
    # Values within rows
    xs = [row["x"] for row in evens]
    assert xs == [0, 2, 4]

    # Use condvars
    under = list(df.where("x < threshold", condvars={"threshold": 3}))
    assert len(under) == 3
    xs_under = [row["x"] for row in under]
    assert xs_under == [0, 1, 2]


def test_sortby_inplace_and_copy():
    df = make_df(6)
    # Shuffle
    perm = np.array([3, 1, 5, 2, 4, 0])
    for k in df.keys():
        df[k] = df[k][perm]  # pyright: ignore

    # In-place ascending sort by x
    out = df.sortby("x", reverse=False, copy=False)
    assert out is None
    np.testing.assert_array_equal(df["x"], np.arange(6))

    # Copy descending sort by y
    df2 = df.sortby("y", reverse=True, copy=True)
    assert isinstance(df2, DictDataFrame)
    np.testing.assert_array_equal(df2["y"], np.arange(6)[::-1] ** 2)
    # Original unchanged ordering for y (still ascending from prior in-place sort)
    np.testing.assert_array_equal(df["y"], np.arange(6) ** 2)


def test_join_with_prefix_and_missing_values():
    left = DictDataFrame(id=np.arange(6), val=np.arange(6) * 10.0)
    right = DictDataFrame(id=np.array([0, 2, 4]), feat=np.array([100, 200, 300]))

    # Join with prefix to avoid name collisions and to skip buggy prefix=None path
    left.join("id", right, key_other="id", columns_other=["feat"], prefix="r_")

    # Expect r_feat populated for ids 0,2,4, and NaN where missing (1,3,5)
    assert "r_feat" in left
    expected = np.array([100.0, np.nan, 200.0, np.nan, 300.0, np.nan])
    np.testing.assert_allclose(left["r_feat"], expected, equal_nan=True)  # pyright: ignore


def test_evalexpr_column_math_and_vars():
    df = make_df(5)

    # Column math
    w = df.evalexpr("x + y")
    np.testing.assert_array_equal(w, df["x"] + df["y"])  # pyright: ignore

    # With external variables
    off = df.evalexpr("x + offset", exprvars={"offset": 10})
    np.testing.assert_array_equal(off, df["x"] + 10)  # pyright: ignore

    # Use numpy functions available via evalexpr
    hyp = df.evalexpr("np.hypot(x, z)", exprvars={"np": np})
    np.testing.assert_allclose(hyp, np.hypot(df["x"], df["z"]))  # pyright: ignore


def test_iterlines_and_rows_alias():
    df = make_df(4)
    rows = list(df.iterlines())
    assert len(rows) == df.nrows
    # rows and .rows alias produce same
    rows2 = list(df.rows)
    assert len(rows2) == df.nrows
    # Check a couple of values
    assert rows[1]["x"] == 1
    assert rows[3]["s"] == "d"
