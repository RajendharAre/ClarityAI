from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.models.analysis import AnalysisRecord


def save_analysis_record(session: Session, filename: str, result: Dict[str, Any]) -> AnalysisRecord:
    if not result:
        result = {}

    record = AnalysisRecord(
        filename=filename,
        quality_label=result.get("quality_label", "UNKNOWN"),
        quality_score=float(result.get("quality_score", 0.0)),
        confidence=float(result.get("confidence", 0.0)),
        anomaly_score=float(result.get("anomaly_score", 0.0)),
        result_json=result,
        created_at=datetime.utcnow(),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def get_analysis_history(session: Session) -> List[AnalysisRecord]:
    return session.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).all()


def get_analysis_by_id(session: Session, analysis_id: int) -> AnalysisRecord | None:
    return session.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
