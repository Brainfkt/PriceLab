import pandas as pd
import pytest

from pricelab.cli import main
from pricelab.data.importers import standardize_columns
from pricelab.data.mapping import infer_column_mapping, missing_required_columns


def test_mapping_infers_common_aliases_and_standardizes():
    raw = pd.DataFrame(
        {
            "order_date": ["2025-01-01"],
            "sku": ["A"],
            "name": ["Widget"],
            "department": ["Tools"],
            "qty": [5],
            "unit_price": [10.0],
            "cogs": [4.0],
            "inventory": [12],
            "promo": ["yes"],
        }
    )
    mapping = infer_column_mapping(raw.columns)
    assert missing_required_columns(mapping) == []
    df = standardize_columns(raw, mapping)
    assert list(df["product_id"]) == ["A"]
    assert bool(df["promotion_flag"].iloc[0]) is True
    assert "channel" in df.columns
    assert df["channel"].iloc[0] == "All"


def test_standardize_rejects_missing_required_mappings():
    raw = pd.DataFrame({"date": ["2025-01-01"], "qty": [5], "unit_price": [10.0]})
    with pytest.raises(ValueError, match="Missing required mappings"):
        standardize_columns(raw)


def test_cli_validate_fails_on_missing_required_schema(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2025-01-01"], "qty": [5], "unit_price": [10.0]}).to_csv(path, index=False)
    exit_code = main(["validate", str(path)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing_required_mappings" in captured.out
