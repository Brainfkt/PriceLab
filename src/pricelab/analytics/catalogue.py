from __future__ import annotations

import pandas as pd

from pricelab.modeling.reliability import compute_reliability


def catalogue_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    revenue = float((df["units_sold"] * df["price"]).sum())
    margin = float(((df["price"] - df["cost"]) * df["units_sold"]).sum()) if "cost" in df.columns else 0.0
    units = float(df["units_sold"].sum())
    return {
        "products": int(df["product_id"].nunique()),
        "rows": int(len(df)),
        "units": round(units, 2),
        "revenue": round(revenue, 2),
        "gross_margin": round(margin, 2),
        "avg_price": round(float(df["price"].mean()), 2),
        "promo_rate": round(float(df["promotion_flag"].astype(bool).mean()), 3) if "promotion_flag" in df.columns else 0.0,
    }


def filter_catalogue(
    df: pd.DataFrame,
    categories: list[str] | None = None,
    channels: list[str] | None = None,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    frame = df.copy()
    filters = {
        "category": categories,
        "channel": channels,
        "region": regions,
    }
    for column, values in filters.items():
        if values and column in frame.columns:
            frame = frame[frame[column].astype(str).isin([str(value) for value in values])]
    return frame


def product_leaderboard(
    df: pd.DataFrame,
    metric: str = "revenue",
    top_n: int | None = 15,
    filters: dict[str, list[str] | None] | None = None,
) -> pd.DataFrame:
    frame = filter_catalogue(
        df,
        categories=filters.get("category") if filters else None,
        channels=filters.get("channel") if filters else None,
        regions=filters.get("region") if filters else None,
    )
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "category",
                "units",
                "revenue",
                "gross_margin",
                "margin_rate",
                "avg_price",
                "price_points",
            ]
        )
    if "promotion_flag" not in frame.columns:
        frame["promotion_flag"] = False
    if "stockout_flag" not in frame.columns:
        frame["stockout_flag"] = frame["stock_available"] <= 0 if "stock_available" in frame.columns else False
    frame["revenue"] = frame["units_sold"] * frame["price"]
    frame["gross_margin"] = (frame["price"] - frame["cost"]) * frame["units_sold"] if "cost" in frame.columns else 0.0
    agg = (
        frame.groupby(["product_id", "product_name", "category"], dropna=False)
        .agg(
            units=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
            avg_price=("price", "mean"),
            price_points=("price", "nunique"),
            promo_rate=("promotion_flag", "mean"),
            stockout_rate=("stockout_flag", "mean"),
        )
        .reset_index()
    )
    agg["margin_rate"] = (agg["gross_margin"] / agg["revenue"].replace(0, pd.NA)).fillna(0.0)
    metric = metric if metric in agg.columns else "revenue"
    result = agg.sort_values(metric, ascending=False).reset_index(drop=True)
    if top_n is not None:
        result = result.head(top_n)
    return result.reset_index(drop=True)


def portfolio_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "units", "revenue", "gross_margin"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["revenue"] = frame["units_sold"] * frame["price"]
    frame["gross_margin"] = (frame["price"] - frame["cost"]) * frame["units_sold"] if "cost" in frame.columns else 0.0
    return (
        frame.dropna(subset=["date"])
        .groupby("date", as_index=False)
        .agg(units=("units_sold", "sum"), revenue=("revenue", "sum"), gross_margin=("gross_margin", "sum"))
        .sort_values("date")
    )


def category_mix(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if frame.empty or "category" not in frame.columns:
        return pd.DataFrame(columns=["category", "units", "revenue", "gross_margin", "margin_rate", "products"])
    frame["revenue"] = frame["units_sold"] * frame["price"]
    frame["gross_margin"] = (frame["price"] - frame["cost"]) * frame["units_sold"] if "cost" in frame.columns else 0.0
    result = (
        frame.groupby("category", dropna=False)
        .agg(
            products=("product_id", "nunique"),
            units=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    result["margin_rate"] = (result["gross_margin"] / result["revenue"].replace(0, pd.NA)).fillna(0.0)
    return result.reset_index(drop=True)


def product_pareto(df: pd.DataFrame, metric: str = "revenue", top_n: int | None = None) -> pd.DataFrame:
    leaderboard = product_leaderboard(df, metric=metric, top_n=top_n)
    if leaderboard.empty:
        return leaderboard.assign(cumulative_share=pd.Series(dtype=float))
    metric = metric if metric in leaderboard.columns else "revenue"
    total = float(leaderboard[metric].sum())
    leaderboard["cumulative_share"] = leaderboard[metric].cumsum() / total if total > 0 else 0.0
    return leaderboard


def portfolio_health(df: pd.DataFrame, product_backtest_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if df.empty or "product_id" not in df.columns:
        return pd.DataFrame(columns=["product_id", "product_name", "category", "reliability_score", "reliability_level", "promo_rate", "stockout_rate"])
    for product_id, product in df.groupby("product_id", dropna=False):
        product_id_str = str(product_id)
        reliability = compute_reliability(df, product_id_str, product_backtest_metrics=product_backtest_metrics)
        rows.append(
            {
                "product_id": product_id_str,
                "product_name": str(product["product_name"].mode().iloc[0]) if "product_name" in product.columns else product_id_str,
                "category": str(product["category"].mode().iloc[0]) if "category" in product.columns else "Unknown",
                "reliability_score": reliability.score,
                "reliability_level": reliability.level,
                "promo_rate": float(product.get("promotion_flag", pd.Series(False, index=product.index)).astype(bool).mean()),
                "stockout_rate": float((product.get("stock_available", pd.Series(1, index=product.index)) <= 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["reliability_score", "product_id"], ascending=[True, True]).reset_index(drop=True)
