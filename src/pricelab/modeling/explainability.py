from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from pricelab.modeling.demand_model import DemandModelBundle
from pricelab.modeling.elasticity import ElasticityResult


def elasticity_coefficient_table(result: ElasticityResult) -> pd.DataFrame:
    rows = [
        {"feature": feature, "coefficient": coefficient, "abs_coefficient": abs(coefficient)}
        for feature, coefficient in result.coefficients.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
    return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False)


def permutation_importance_table(
    bundle: DemandModelBundle,
    df: pd.DataFrame,
    max_rows: int = 600,
    random_state: int = 42,
) -> pd.DataFrame:
    features = bundle.numeric_features + bundle.categorical_features
    sample = df.dropna(subset=["units_sold"]).copy()
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=random_state)
    X = sample[features]
    y = sample["units_sold"].astype(float)
    result = permutation_importance(
        bundle.model,
        X,
        y,
        n_repeats=5,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
    )
    return (
        pd.DataFrame(
            {
                "feature": features,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

