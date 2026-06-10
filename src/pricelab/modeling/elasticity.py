from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from pricelab.features.build import build_model_frame


@dataclass
class ElasticityResult:
    product_id: str
    elasticity: float
    ci_low: float | None
    ci_high: float | None
    n_obs: int
    price_points: int
    r2_train: float
    model: Ridge | None
    feature_columns: list[str] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


ELASTICITY_NUMERIC_CONTROLS = [
    "log_price",
    "promotion_flag",
    "discount_rate",
    "promo_discount_interaction",
    "competitor_gap",
    "log_marketing_spend",
    "log_traffic",
    "weather_index",
    "holiday_flag",
    "stockout_flag",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
]
ELASTICITY_CATEGORICAL_CONTROLS = ["channel", "region", "customer_segment"]


def fit_loglog_elasticity(
    df: pd.DataFrame,
    product_id: str | None = None,
    alpha: float = 0.05,
    bootstrap_samples: int = 80,
    random_state: int = 42,
) -> ElasticityResult:
    frame = _ensure_feature_frame(df)
    if product_id is None:
        product_id = str(frame["product_id"].iloc[0]) if "product_id" in frame.columns and len(frame) else "catalogue"
    data = frame[frame["product_id"].astype(str) == str(product_id)].copy()
    warnings: list[str] = []

    if data.empty:
        return _empty_result(product_id, "No rows for this product.")

    usable = data[(data["price"] > 0) & (data["units_sold"] >= 0)].copy()
    if "stockout_flag" in usable.columns:
        without_stockouts = usable[~usable["stockout_flag"].astype(bool)].copy()
        if len(without_stockouts) >= 12:
            usable = without_stockouts
        else:
            warnings.append("Stockout-free sample is small; model keeps censored rows.")

    n_obs = int(len(usable))
    price_points = int(usable["price"].nunique()) if "price" in usable.columns else 0
    if n_obs < 8 or price_points < 2:
        return _empty_result(product_id, "Not enough observations or price variation.", n_obs, price_points)

    X = _design_matrix(usable)
    y = usable["log_units"].astype(float).to_numpy()
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    pred = model.predict(X)
    r2 = float(r2_score(y, pred)) if len(np.unique(y)) > 1 else np.nan
    coefficients = {col: float(value) for col, value in zip(X.columns, model.coef_)}
    elasticity = float(coefficients.get("log_price", np.nan))

    if price_points < 3:
        warnings.append("Elasticity is weak because fewer than 3 prices are observed.")
    if elasticity > 0:
        warnings.append("Estimated elasticity is positive; this may reflect confounding or promotions.")

    ci_low, ci_high = _bootstrap_elasticity(
        usable,
        feature_columns=list(X.columns),
        alpha=alpha,
        samples=bootstrap_samples,
        random_state=random_state,
    )
    return ElasticityResult(
        product_id=str(product_id),
        elasticity=elasticity,
        ci_low=ci_low,
        ci_high=ci_high,
        n_obs=n_obs,
        price_points=price_points,
        r2_train=r2,
        model=model,
        feature_columns=list(X.columns),
        coefficients=coefficients,
        warnings=warnings,
    )


def fit_catalogue_elasticities(df: pd.DataFrame, min_obs: int = 8) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = _ensure_feature_frame(df)
    for product_id, product_df in frame.groupby("product_id"):
        if len(product_df) < min_obs:
            continue
        result = fit_loglog_elasticity(product_df, str(product_id), bootstrap_samples=20)
        rows.append(
            {
                "product_id": str(product_id),
                "elasticity": result.elasticity,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "n_obs": result.n_obs,
                "price_points": result.price_points,
                "r2_train": result.r2_train,
                "warnings": "; ".join(result.warnings),
            }
        )
    return pd.DataFrame(rows)


def predict_units_from_elasticity(result: ElasticityResult, df: pd.DataFrame) -> np.ndarray:
    if result.model is None:
        return np.full(len(df), np.nan)
    frame = _ensure_feature_frame(df)
    X = _design_matrix(frame).reindex(columns=result.feature_columns, fill_value=0.0)
    pred_log = result.model.predict(X)
    return np.expm1(pred_log).clip(min=0)


def _ensure_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"log_price", "log_units", "week_sin"}
    if required.issubset(df.columns):
        return df.copy()
    return build_model_frame(df, weekly=False)


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    numeric = [col for col in ELASTICITY_NUMERIC_CONTROLS if col in frame.columns]
    categorical = [col for col in ELASTICITY_CATEGORICAL_CONTROLS if col in frame.columns]
    X_num = frame[numeric].copy()
    for col in X_num.columns:
        if X_num[col].dtype == bool:
            X_num[col] = X_num[col].astype(float)
    X_num = X_num.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if categorical:
        X_cat = pd.get_dummies(frame[categorical].fillna("Unknown").astype(str), drop_first=True, dtype=float)
        return pd.concat([X_num, X_cat], axis=1)
    return X_num


def _bootstrap_elasticity(
    df: pd.DataFrame,
    feature_columns: list[str],
    alpha: float,
    samples: int,
    random_state: int,
) -> tuple[float | None, float | None]:
    if len(df) < 18 or samples <= 0:
        return None, None
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    y_all = df["log_units"].astype(float).to_numpy()
    for _ in range(samples):
        idx = rng.integers(0, len(df), len(df))
        sample = df.iloc[idx].copy()
        X = _design_matrix(sample).reindex(columns=feature_columns, fill_value=0.0)
        if X["log_price"].nunique() <= 1:
            continue
        model = Ridge(alpha=alpha)
        try:
            model.fit(X, y_all[idx])
            values.append(float(model.coef_[feature_columns.index("log_price")]))
        except Exception:
            continue
    if len(values) < 10:
        return None, None
    return float(np.percentile(values, 5)), float(np.percentile(values, 95))


def _empty_result(
    product_id: str,
    warning: str,
    n_obs: int = 0,
    price_points: int = 0,
) -> ElasticityResult:
    return ElasticityResult(
        product_id=str(product_id),
        elasticity=float("nan"),
        ci_low=None,
        ci_high=None,
        n_obs=n_obs,
        price_points=price_points,
        r2_train=float("nan"),
        model=None,
        warnings=[warning],
    )
