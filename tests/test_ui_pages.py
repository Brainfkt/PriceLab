from streamlit.testing.v1 import AppTest


def test_opportunities_page_handles_small_catalogue():
    script = """
import pandas as pd
from pricelab.ui.pages.opportunities import render_opportunities_page

df = pd.DataFrame(
    {
        "date": pd.date_range("2025-01-01", periods=2, freq="W"),
        "product_id": ["A", "B"],
        "product_name": ["Widget A", "Widget B"],
        "category": ["Tools", "Tools"],
        "channel": ["Online", "Online"],
        "region": ["North", "North"],
        "units_sold": [10.0, 12.0],
        "price": [10.0, 12.0],
        "cost": [5.0, 6.0],
        "stock_available": [50.0, 50.0],
        "promotion_flag": [False, False],
        "discount_rate": [0.0, 0.0],
    }
)
render_opportunities_page(df, "revenue", None)
"""
    app = AppTest.from_string(script)
    app.run(timeout=30)
    assert not app.exception
