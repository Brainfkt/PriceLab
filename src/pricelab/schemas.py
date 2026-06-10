from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class PricingObjective(str, Enum):
    VOLUME = "volume"
    REVENUE = "revenue"
    MARGIN = "margin"
    PRUDENCE = "prudence"


class ColumnMapping(BaseModel):
    date: str | None = None
    product_id: str | None = None
    product_name: str | None = None
    category: str | None = None
    units_sold: str | None = None
    price: str | None = None
    cost: str | None = None
    stock_available: str | None = None
    promotion_flag: str | None = None
    discount_rate: str | None = None
    channel: str | None = None
    region: str | None = None
    competitor_price: str | None = None
    marketing_spend: str | None = None
    traffic: str | None = None
    holiday_flag: str | None = None
    customer_segment: str | None = None
    returns: str | None = None
    weather_index: str | None = None

    def as_dict(self) -> dict[str, str]:
        data = self.model_dump() if hasattr(self, "model_dump") else self.dict()
        return {k: v for k, v in data.items() if v}


class DataQualityIssue(BaseModel):
    severity: Severity
    code: str
    message: str
    metric: float | int | str | None = None
    product_id: str | None = None


class DataQualityReport(BaseModel):
    row_count: int
    product_count: int
    start_date: str | None = None
    end_date: str | None = None
    issues: list[DataQualityIssue] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == Severity.WARNING)


class ReliabilityResult(BaseModel):
    product_id: str
    score: float
    level: str
    components: dict[str, float]
    reasons: list[str] = Field(default_factory=list)
    hard_blocks: list[str] = Field(default_factory=list)

    @property
    def allows_recommendation(self) -> bool:
        return self.score >= 55 and not self.hard_blocks

    @property
    def allows_simulation(self) -> bool:
        return self.score >= 35 and not self.hard_blocks


class ScenarioResult(BaseModel):
    product_id: str
    price: float
    reference_price: float
    predicted_units: float
    predicted_revenue: float
    predicted_margin: float | None = None
    low_units: float
    high_units: float
    reliability_score: float
    status: str
    warnings: list[str] = Field(default_factory=list)


class PriceRecommendation(BaseModel):
    product_id: str
    objective: PricingObjective
    status: str
    recommended_price: float | None = None
    lower_price: float | None = None
    upper_price: float | None = None
    expected_units: float | None = None
    expected_revenue: float | None = None
    expected_margin: float | None = None
    reliability_score: float
    reasons: list[str] = Field(default_factory=list)
