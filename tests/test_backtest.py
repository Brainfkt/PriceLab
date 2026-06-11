import pandas as pd

from pricelab.data.demo_generator import generate_demo_dataset
from pricelab.data.importers import standardize_columns
from pricelab.features.build import build_model_frame
from pricelab.modeling.backtest import make_time_splits, run_backtest


def test_time_splits_are_ordered():
    dates = list(pd.date_range("2025-01-01", periods=30, freq="W"))
    splits = make_time_splits(dates, n_splits=3)
    assert splits
    for train_end, test_start, test_end in splits:
        assert train_end < test_start <= test_end


def test_run_backtest_returns_metrics_on_demo_subset():
    df = standardize_columns(generate_demo_dataset(seed=3, n_products=4, periods=24))
    frame = build_model_frame(df, weekly=True)
    result = run_backtest(frame, n_splits=2)
    assert result.valid
    assert result.metrics["wmape"] >= 0
    assert result.metrics["smape"] >= 0
    assert not result.product_metrics.empty
    assert not result.predictions.empty
    assert {"date", "actual", "predicted", "baseline"}.issubset(result.predictions.columns)
