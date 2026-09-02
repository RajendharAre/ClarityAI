import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.core.config import get_settings
from app.db.database import SessionLocal, init_db
from app.services.analysis_service import save_analysis_record
from ml.inference import infer

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="ClarityAI image quality and defect detection API",
)

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.post("/api/v1/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed = set(settings.allowed_formats_list)
    content_type = file.content_type or ""
    if content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    destination = upload_dir / f"{uuid4().hex[:12]}_{file.filename}"
    contents = await file.read()
    if len(contents) > settings.max_file_size:
        raise HTTPException(status_code=400, detail="File exceeds max upload size")

    destination.write_bytes(contents)

    try:
        result = infer(str(destination))
    except Exception as exc:  # pragma: no cover - handled by API contract
        raise HTTPException(status_code=400, detail=f"Image analysis failed: {exc}") from exc

    with SessionLocal() as db:
        record = save_analysis_record(db, file.filename, result)
        result["analysis_id"] = record.id
        result["filename"] = file.filename
        result["created_at"] = record.created_at.isoformat()

    return result


@app.get("/api/v1/history")
async def list_history():
    with SessionLocal() as db:
        from app.models.analysis import AnalysisRecord
        records = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).all()

    return [
        {
            "analysis_id": record.id,
            "filename": record.filename,
            "quality_label": record.quality_label,
            "quality_score": record.quality_score,
            "confidence": record.confidence,
            "anomaly_score": record.anomaly_score,
            "created_at": record.created_at.isoformat(),
        }
        for record in records
    ]


@app.get("/api/v1/history/{analysis_id}")
async def get_history_item(analysis_id: int):
    with SessionLocal() as db:
        from app.models.analysis import AnalysisRecord
        record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()

    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "analysis_id": record.id,
        "filename": record.filename,
        "quality_label": record.quality_label,
        "quality_score": record.quality_score,
        "confidence": record.confidence,
        "anomaly_score": record.anomaly_score,
        "created_at": record.created_at.isoformat(),
        "result": record.result_json,
    }


@app.get("/")
async def root():
    return {"message": "ClarityAI API is running", "docs": "/docs"}
