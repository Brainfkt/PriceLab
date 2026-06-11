import numpy as np
import pandas as pd

from pricelab.modeling.elasticity import fit_loglog_elasticity
from pricelab.modeling.optimization import find_price_recommendation
from pricelab.modeling.reliability import compute_reliability
import plotly.graph_objects as go

from pricelab.reporting.html import build_html_report_with_figures, markdown_to_basic_html
from pricelab.reporting.markdown import build_product_markdown_report


def test_markdown_and_html_report_contain_core_sections():
    prices = np.linspace(8, 16, 20)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20, freq="W"),
            "product_id": ["A"] * 20,
            "product_name": ["Widget"] * 20,
            "category": ["Tools"] * 20,
            "channel": ["Online"] * 20,
            "region": ["North"] * 20,
            "units_sold": 100 * (prices / 10) ** -1.1,
            "price": prices,
            "cost": [5.0] * 20,
            "stock_available": [999.0] * 20,
            "promotion_flag": [False] * 20,
        }
    )
    reliability = compute_reliability(df, "A")
    recommendation = find_price_recommendation(df, "A", "revenue")
    elasticity = fit_loglog_elasticity(df, "A", bootstrap_samples=0)
    markdown = build_product_markdown_report(df, "A", reliability, recommendation, elasticity)
    html = markdown_to_basic_html(markdown)
    rich_html = build_html_report_with_figures(markdown, [("Chart", go.Figure())])
    assert "Executive Summary" in markdown
    assert "Reliability Drivers" in markdown
    assert "<html>" in html
    assert "plotly" in rich_html.lower()
