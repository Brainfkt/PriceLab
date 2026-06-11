from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.data.demo_generator import generate_demo_dataset
from pricelab.data.importers import standardize_columns
from pricelab.data.mapping import infer_column_mapping
from pricelab.features.build import build_model_frame
from pricelab.schemas import ColumnMapping


@st.cache_data(show_spinner=False)
def demo_data() -> pd.DataFrame:
    return generate_demo_dataset()


@st.cache_data(show_spinner=False)
def standardize_cached(raw: pd.DataFrame, mapping_dict: dict[str, str | None]) -> pd.DataFrame:
    return standardize_columns(raw, ColumnMapping(**mapping_dict))


@st.cache_data(show_spinner=False)
def feature_frame_cached(df: pd.DataFrame) -> pd.DataFrame:
    return build_model_frame(df, weekly="auto")


def default_mapping(raw: pd.DataFrame) -> ColumnMapping:
    return infer_column_mapping(raw.columns)
