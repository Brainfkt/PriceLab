from pricelab.data.demo_generator import generate_demo_dataset
from pricelab.data.importers import standardize_columns
from pricelab.features.build import build_model_frame


def test_feature_frame_adds_business_and_time_features():
    df = standardize_columns(generate_demo_dataset(seed=1, n_products=2, periods=8))
    frame = build_model_frame(df, weekly=True)
    for column in ["log_price", "log_units", "revenue", "gross_margin", "week_sin", "rolling_units_4"]:
        assert column in frame.columns
    assert frame["rolling_units_4"].isna().sum() == 0
    assert frame["revenue"].sum() > 0

