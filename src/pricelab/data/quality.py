from __future__ import annotations

import pandas as pd

from pricelab.config import KEY_COLUMNS, REQUIRED_COLUMNS, THRESHOLDS
from pricelab.schemas import DataQualityIssue, DataQualityReport, Severity


def scan_quality(df: pd.DataFrame) -> DataQualityReport:
    issues: list[DataQualityIssue] = []
    metrics: dict[str, float | int | str] = {}
    row_count = int(len(df))
    product_count = int(df["product_id"].nunique()) if "product_id" in df.columns else 0

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    for col in missing_columns:
        issues.append(
            DataQualityIssue(
                severity=Severity.ERROR,
                code="missing_required_column",
                message=f"Required column is missing: {col}",
                metric=col,
            )
        )

    if row_count == 0:
        issues.append(
            DataQualityIssue(
                severity=Severity.ERROR,
                code="empty_dataset",
                message="Dataset has no rows.",
            )
        )
        return DataQualityReport(row_count=0, product_count=0, issues=issues, metrics=metrics)

    start_date = _date_string(df["date"].min()) if "date" in df.columns else None
    end_date = _date_string(df["date"].max()) if "date" in df.columns else None

    if "date" in df.columns:
        invalid_dates = int(df["date"].isna().sum())
        metrics["invalid_dates"] = invalid_dates
        if invalid_dates:
            issues.append(
                DataQualityIssue(
                    severity=Severity.ERROR,
                    code="invalid_dates",
                    message="Some rows have invalid dates.",
                    metric=invalid_dates,
                )
            )

    for col in ["units_sold", "price", "cost", "stock_available"]:
        if col in df.columns:
            missing_rate = float(df[col].isna().mean())
            metrics[f"{col}_missing_rate"] = missing_rate
            if missing_rate > 0.0:
                severity = Severity.ERROR if col in {"units_sold", "price"} else Severity.WARNING
                issues.append(
                    DataQualityIssue(
                        severity=severity,
                        code=f"{col}_missing",
                        message=f"{col} has missing values.",
                        metric=round(missing_rate, 3),
                    )
                )

    if "price" in df.columns:
        bad_price = int(((df["price"] <= 0) | df["price"].isna()).sum())
        metrics["non_positive_price_rows"] = bad_price
        if bad_price:
            issues.append(
                DataQualityIssue(
                    severity=Severity.ERROR,
                    code="non_positive_price",
                    message="Price must be strictly positive.",
                    metric=bad_price,
                )
            )

    for col in ["units_sold", "cost", "stock_available", "returns"]:
        if col in df.columns:
            negative = int((df[col] < 0).sum())
            metrics[f"negative_{col}_rows"] = negative
            if negative:
                issues.append(
                    DataQualityIssue(
                        severity=Severity.ERROR,
                        code=f"negative_{col}",
                        message=f"{col} contains negative values.",
                        metric=negative,
                    )
                )

    duplicate_cols = [col for col in KEY_COLUMNS if col in df.columns]
    if len(duplicate_cols) == len(KEY_COLUMNS):
        duplicate_rows = int(df.duplicated(duplicate_cols).sum())
        metrics["duplicate_key_rows"] = duplicate_rows
        if duplicate_rows:
            issues.append(
                DataQualityIssue(
                    severity=Severity.WARNING,
                    code="duplicate_keys",
                    message="Duplicate date/product/channel/region rows were detected.",
                    metric=duplicate_rows,
                )
            )

    if "stock_available" in df.columns:
        stockout_rate = float((df["stock_available"] <= 0).mean())
        metrics["stockout_rate"] = stockout_rate
        if stockout_rate >= THRESHOLDS.bad_stockout_rate:
            issues.append(
                DataQualityIssue(
                    severity=Severity.WARNING,
                    code="high_stockout_rate",
                    message="Stockouts are frequent enough to censor demand.",
                    metric=round(stockout_rate, 3),
                )
            )

    if "promotion_flag" in df.columns:
        promo_rate = float(df["promotion_flag"].astype(bool).mean())
        metrics["promotion_rate"] = promo_rate
        if promo_rate >= THRESHOLDS.bad_promo_rate:
            issues.append(
                DataQualityIssue(
                    severity=Severity.WARNING,
                    code="promo_contamination",
                    message="Promotions dominate the history and may contaminate elasticity.",
                    metric=round(promo_rate, 3),
                )
            )

    if {"product_id", "price", "date"}.issubset(df.columns):
        product_stats = (
            df.groupby("product_id")
            .agg(
                observations=("date", "count"),
                first_date=("date", "min"),
                last_date=("date", "max"),
                price_points=("price", "nunique"),
                price_cv=("price", lambda s: float(s.std(ddof=0) / s.mean()) if s.mean() else 0.0),
            )
            .reset_index()
        )
        fragile = product_stats[
            (product_stats["price_points"] < THRESHOLDS.min_price_points)
            | (product_stats["price_cv"] <= THRESHOLDS.low_price_cv)
        ]
        metrics["products_with_low_price_variation"] = int(len(fragile))
        for _, row in fragile.head(20).iterrows():
            issues.append(
                DataQualityIssue(
                    severity=Severity.WARNING,
                    code="low_price_variation",
                    message="Product has too few price changes for robust elasticity.",
                    metric=f"{int(row['price_points'])} prices, cv={row['price_cv']:.3f}",
                    product_id=str(row["product_id"]),
                )
            )

    return DataQualityReport(
        row_count=row_count,
        product_count=product_count,
        start_date=start_date,
        end_date=end_date,
        issues=issues,
        metrics=metrics,
    )


def quality_score(report: DataQualityReport) -> float:
    score = 100.0
    score -= 25.0 * report.error_count
    score -= 7.0 * report.warning_count
    return float(max(0.0, min(100.0, score)))


def _date_string(value: object) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()

