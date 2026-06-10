from __future__ import annotations

import pandas as pd


def best_moments(df: pd.DataFrame, product_id: str, metric: str = "revenue") -> pd.DataFrame:
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    if product.empty:
        return pd.DataFrame()
    product["date"] = pd.to_datetime(product["date"])
    product["month"] = product["date"].dt.month
    product["revenue"] = product["units_sold"] * product["price"]
    product["gross_margin"] = (product["price"] - product["cost"]) * product["units_sold"] if "cost" in product.columns else 0.0
    group_cols = ["month", "channel", "region"]
    metric = metric if metric in {"units_sold", "revenue", "gross_margin"} else "revenue"
    return (
        product.groupby(group_cols, dropna=False)
        .agg(
            observations=("date", "count"),
            units=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
            avg_price=("price", "mean"),
        )
        .reset_index()
        .sort_values(metric if metric != "units_sold" else "units", ascending=False)
        .head(20)
    )

