from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pricelab.features.build import build_model_frame, model_feature_columns


@dataclass
class DemandModelBundle:
    model: Pipeline
    numeric_features: list[str]
    categorical_features: list[str]
    target: str = "log_units"


def train_demand_model(df: pd.DataFrame, random_state: int = 42) -> DemandModelBundle:
    frame = _ensure_feature_frame(df)
    numeric, categorical = model_feature_columns(frame)
    features = numeric + categorical
    train = frame.dropna(subset=["units_sold", "price"]).copy()
    X = train[features]
    y = np.log1p(train["units_sold"].clip(lower=0).astype(float))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return DemandModelBundle(model=model, numeric_features=numeric, categorical_features=categorical)


def predict_demand(bundle: DemandModelBundle, df: pd.DataFrame) -> np.ndarray:
    frame = _ensure_feature_frame(df)
    features = bundle.numeric_features + bundle.categorical_features
    for col in features:
        if col not in frame.columns:
            frame[col] = np.nan
    pred_log = bundle.model.predict(frame[features])
    return np.expm1(pred_log).clip(min=0)


def _ensure_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_features = {"log_price", "week_sin", "rolling_units_4"}
    if required_features.issubset(df.columns):
        return df.copy()
    return build_model_frame(df, weekly=False)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

