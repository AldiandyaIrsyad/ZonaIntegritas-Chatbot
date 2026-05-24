import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from main import app
from src.knowledge_base.dependency import get_pdf_service

client = TestClient(app)

@pytest.fixture
def mock_kb_service():
    mock_service = AsyncMock()
    app.dependency_overrides[get_pdf_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.pop(get_pdf_service, None)

def test_admin_page():
    response = client.get("/admin/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_get_pdfs(mock_kb_service):
    mock_kb_service.list_pdfs.return_value = [{"id": "1", "title": "Test"}]
    response = client.get("/api/admin/pdfs")
    assert response.status_code == 200
    assert response.json() == [{"id": "1", "title": "Test"}]

def test_upload_pdf(mock_kb_service):
    mock_pdf = MagicMock()
    mock_pdf.id = "1"
    mock_pdf.title = "Test PDF"
    mock_pdf.ingestion_status = "pending"
    mock_kb_service.upload_pdf.return_value = mock_pdf
    
    files = {'file': ('test.pdf', b'dummy content', 'application/pdf')}
    data = {'title': 'Test PDF', 'description': 'Test desc'}
    
    response = client.post("/api/admin/pdfs", data=data, files=files)
    
    assert response.status_code == 202
    assert response.json() == {
        "id": "1",
        "title": "Test PDF",
        "ingestion_status": "pending",
        "status": "accepted"
    }

def test_update_pdf_status(mock_kb_service):
    mock_pdf = MagicMock()
    mock_pdf.id = "1"
    mock_pdf.active = True
    mock_kb_service.update_pdf_status.return_value = mock_pdf
    
    response = client.put("/api/admin/pdfs/1/status", json={"active": True})
    
    assert response.status_code == 200
    assert response.json() == {"id": "1", "active": True, "status": "success"}

def test_update_pdf_status_not_found(mock_kb_service):
    mock_kb_service.update_pdf_status.return_value = None
    response = client.put("/api/admin/pdfs/1/status", json={"active": True})
    assert response.status_code == 404

def test_delete_pdf(mock_kb_service):
    mock_kb_service.delete_pdf.return_value = True
    response = client.delete("/api/admin/pdfs/1")
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "PDF deleted"}

def test_delete_pdf_not_found(mock_kb_service):
    mock_kb_service.delete_pdf.return_value = False
    response = client.delete("/api/admin/pdfs/1")
    assert response.status_code == 404

def test_get_ingestion_status(mock_kb_service):
    mock_kb_service.get_ingestion_status.return_value = {
        "id": "1", "title": "Test PDF", "ingestion_status": "completed"
    }
    response = client.get("/api/admin/pdfs/1/ingestion-status")
    assert response.status_code == 200
    assert response.json() == {
        "id": "1", "title": "Test PDF", "ingestion_status": "completed"
    }

def test_get_ingestion_status_not_found(mock_kb_service):
    mock_kb_service.get_ingestion_status.return_value = None
    response = client.get("/api/admin/pdfs/1/ingestion-status")
    assert response.status_code == 404
