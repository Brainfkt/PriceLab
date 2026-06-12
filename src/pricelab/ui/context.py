from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pricelab.modeling.backtest import BacktestResult
from pricelab.schemas import ColumnMapping, DataQualityReport


@dataclass
class GlobalFilters:
    categories: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)


@dataclass
class AppContext:
    raw_df: pd.DataFrame
    mapping: ColumnMapping
    standardized_df: pd.DataFrame | None
    quality_report: DataQualityReport | None
    frame: pd.DataFrame | None
    backtest_result: BacktestResult | None
    selected_product: str | None
    objective: str
    filters: GlobalFilters
    data_error: str | None = None

    @property
    def data_ready(self) -> bool:
        return self.standardized_df is not None and self.quality_report is not None and self.quality_report.error_count == 0

    @property
    def modelling_ready(self) -> bool:
        return self.data_ready and self.frame is not None and not self.frame.empty

    @property
    def product_metrics(self) -> pd.DataFrame | None:
        if self.backtest_result is not None and self.backtest_result.valid:
            return self.backtest_result.product_metrics
        return None

    @property
    def filtered_frame(self) -> pd.DataFrame | None:
        if self.frame is None:
            return None
        frame = self.frame.copy()
        for column, values in {
            "category": self.filters.categories,
            "channel": self.filters.channels,
            "region": self.filters.regions,
        }.items():
            if values and column in frame.columns:
                frame = frame[frame[column].astype(str).isin(values)]
        return frame
