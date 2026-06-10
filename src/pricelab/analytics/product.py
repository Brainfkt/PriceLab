from __future__ import annotations

import numpy as np
import pandas as pd


def product_summary(df: pd.DataFrame, product_id: str) -> dict[str, float | int | str]:
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    if product.empty:
        return {}
    revenue = float((product["units_sold"] * product["price"]).sum())
    margin = float(((product["price"] - product["cost"]) * product["units_sold"]).sum()) if "cost" in product.columns else np.nan
    return {
        "product_id": str(product_id),
        "product_name": str(product["product_name"].mode().iloc[0]) if "product_name" in product.columns else str(product_id),
        "category": str(product["category"].mode().iloc[0]) if "category" in product.columns else "Unknown",
        "observations": int(len(product)),
        "first_date": pd.Timestamp(product["date"].min()).date().isoformat(),
        "last_date": pd.Timestamp(product["date"].max()).date().isoformat(),
        "units": round(float(product["units_sold"].sum()), 2),
        "revenue": round(revenue, 2),
        "gross_margin": round(margin, 2) if np.isfinite(margin) else np.nan,
        "avg_price": round(float(product["price"].mean()), 2),
        "min_price": round(float(product["price"].min()), 2),
        "max_price": round(float(product["price"].max()), 2),
        "price_points": int(product["price"].nunique()),
        "promo_rate": round(float(product.get("promotion_flag", pd.Series(False, index=product.index)).astype(bool).mean()), 3),
        "stockout_rate": round(float((product.get("stock_available", pd.Series(1, index=product.index)) <= 0).mean()), 3),
    }


def price_performance_bins(df: pd.DataFrame, product_id: str, bins: int = 8) -> pd.DataFrame:
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    if product.empty or product["price"].nunique() < 2:
        return pd.DataFrame()
    product["price_bin"] = pd.qcut(product["price"], q=min(bins, product["price"].nunique()), duplicates="drop")
    product["revenue"] = product["units_sold"] * product["price"]
    product["gross_margin"] = (product["price"] - product["cost"]) * product["units_sold"] if "cost" in product.columns else np.nan
    return (
        product.groupby("price_bin", observed=True)
        .agg(
            price_min=("price", "min"),
            price_max=("price", "max"),
            avg_price=("price", "mean"),
            units=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
            observations=("price", "count"),
        )
        .reset_index()
    )

