import pandas as pd
import plotly.graph_objects as go

from pricelab.analytics.catalogue import category_mix, portfolio_health, portfolio_timeseries, product_leaderboard, product_pareto
from pricelab.analytics.opportunities import scan_catalogue_opportunities
from pricelab.data.demo_generator import generate_demo_dataset
from pricelab.features.build import build_model_frame
from pricelab.ui.components import (
    category_mix_chart,
    opportunity_matrix_chart,
    pareto_chart,
    portfolio_health_chart,
    portfolio_trend_chart,
    temporal_heatmap_chart,
)


def test_product_leaderboard_supports_all_products_filters_and_pareto():
    frame = build_model_frame(generate_demo_dataset(seed=7, n_products=8, periods=10), weekly="auto")
    all_products = product_leaderboard(frame, metric="revenue", top_n=None)
    assert len(all_products) == frame["product_id"].nunique()

    top_products = product_leaderboard(frame, metric="gross_margin", top_n=3)
    assert len(top_products) == 3
    assert top_products["gross_margin"].is_monotonic_decreasing

    category = str(frame["category"].dropna().iloc[0])
    filtered = product_leaderboard(frame, top_n=None, filters={"category": [category]})
    assert filtered["category"].eq(category).all()

    pareto = product_pareto(frame, metric="revenue", top_n=None)
    assert pareto["cumulative_share"].is_monotonic_increasing
    assert round(float(pareto["cumulative_share"].iloc[-1]), 6) == 1.0


def test_portfolio_chart_helpers_return_figures():
    frame = build_model_frame(generate_demo_dataset(seed=8, n_products=6, periods=8), weekly="auto")
    figures = [
        portfolio_trend_chart(portfolio_timeseries(frame)),
        category_mix_chart(category_mix(frame)),
        portfolio_health_chart(portfolio_health(frame)),
        pareto_chart(product_pareto(frame)),
    ]
    assert all(isinstance(figure, go.Figure) for figure in figures)
    assert all(figure.data for figure in figures)


def test_temporal_heatmap_adapts_to_source_grain():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=14, freq="D"),
            "product_id": ["A"] * 14,
            "product_name": ["Widget"] * 14,
            "category": ["Tools"] * 14,
            "channel": ["Online"] * 14,
            "region": ["North"] * 14,
            "units_sold": [10.0] * 14,
            "price": [10.0] * 14,
            "cost": [5.0] * 14,
            "stock_available": [100.0] * 14,
            "promotion_flag": [False] * 14,
            "discount_rate": [0.0] * 14,
        }
    )
    daily_frame = build_model_frame(daily, weekly=False)
    daily_fig = temporal_heatmap_chart(daily_frame, "A")
    assert daily_fig.layout.xaxis.title.text == "Month"
    assert daily_fig.layout.yaxis.title.text == "Day"

    weekly_frame = build_model_frame(generate_demo_dataset(seed=9, n_products=1, periods=10), weekly="auto")
    weekly_fig = temporal_heatmap_chart(weekly_frame, "P001")
    assert weekly_fig.layout.xaxis.title.text == "Promotion depth"
    assert weekly_fig.layout.yaxis.title.text == "Season"


def test_opportunity_matrix_chart_uses_reliability_and_opportunity_scores():
    frame = build_model_frame(generate_demo_dataset(seed=10, n_products=5, periods=14), weekly="auto")
    opportunities = scan_catalogue_opportunities(frame, limit=5)
    figure = opportunity_matrix_chart(opportunities)
    assert isinstance(figure, go.Figure)
    assert {"reliability_score", "opportunity_score"}.issubset(opportunities.columns)
    assert figure.data
