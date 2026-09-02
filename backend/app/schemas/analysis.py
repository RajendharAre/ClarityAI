from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    analysis_id: int
    filename: str
    quality_label: str
    quality_score: float
    confidence: float
    anomaly_score: float
    created_at: datetime


class AnalysisDetailResponse(AnalysisResponse):
    result: Optional[Dict[str, Any]] = None
