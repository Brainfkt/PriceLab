import pandas as pd

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

