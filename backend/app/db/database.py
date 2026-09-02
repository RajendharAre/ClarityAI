from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings

settings = get_settings()
resolved_database_url = settings.resolved_database_url

base_dir = Path(__file__).resolve().parents[2]
if resolved_database_url.startswith("sqlite"):
    db_path = resolved_database_url.replace("sqlite:///", "")
    if not db_path.startswith("/"):
        db_path = str(base_dir / db_path)
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    resolved_database_url = f"sqlite:///{db_path_obj.as_posix()}"

engine = create_engine(
    resolved_database_url,
    connect_args={"check_same_thread": False} if resolved_database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

from app.models.analysis import AnalysisRecord  # noqa: E402


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
