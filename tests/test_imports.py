import importlib


def test_import_simpletable_top_level():
    mod = importlib.import_module("simpletable")
    assert hasattr(mod, "__all__")
    # Ensure public symbols are exposed
    for name in ("DataFrame", "DictDataFrame"):
        assert name in mod.__all__
        assert getattr(mod, name) is not None


def test_import_public_symbols():
    from simpletable import DataFrame, DictDataFrame  # noqa: F401

    # If imports succeed, that's enough for this smoke test
    assert True
