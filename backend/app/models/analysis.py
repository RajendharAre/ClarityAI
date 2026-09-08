from datetime import datetime

from sqlalchemy import JSON, String, Float, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    quality_label: Mapped[str] = mapped_column(String(50), nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
