from pricelab.config import OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from pricelab.data.demo_generator import generate_demo_dataset


def test_demo_generator_has_expected_schema_and_shape():
    df = generate_demo_dataset(seed=7, n_products=3, periods=5)
    assert len(df) == 3 * 5 * 2 * 4
    for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        assert column in df.columns
    assert df["price"].min() > 0
    assert df["units_sold"].min() >= 0

