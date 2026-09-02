from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "ClarityAI"


def test_analyze_endpoint_accepts_upload():
    with open("../test_image.jpg", "rb") as image_file:
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["quality_label"] in {"ACCEPTABLE", "DEGRADED", "DEFECTIVE", "UNKNOWN"}
    assert 0 <= payload["quality_score"] <= 100
    assert "analysis_id" in payload
