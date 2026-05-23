import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from main import app
from src.chat.dependency import get_chat_service

client = TestClient(app)

@pytest.fixture
def mock_chat_service():
    mock_service = AsyncMock()
    app.dependency_overrides[get_chat_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.pop(get_chat_service, None)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_get_sessions(mock_chat_service):
    mock_chat_service.list_sessions.return_value = [{"id": "1", "title": "Test"}]
    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert response.json() == [{"id": "1", "title": "Test"}]

def test_create_session(mock_chat_service):
    mock_chat_service.create_new_session.return_value = {"id": "1", "title": "New Chat"}
    response = client.post("/api/sessions")
    assert response.status_code == 200
    assert response.json() == {"id": "1", "title": "New Chat"}

def test_get_session_details(mock_chat_service):
    mock_chat_service.get_session_details.return_value = {"title": "Test", "messages": []}
    response = client.get("/api/sessions/1")
    assert response.status_code == 200
    assert response.json() == {"title": "Test", "messages": []}

def test_get_session_details_not_found(mock_chat_service):
    mock_chat_service.get_session_details.return_value = None
    response = client.get("/api/sessions/1")
    assert response.status_code == 404

def test_delete_session(mock_chat_service):
    mock_chat_service.delete_session.return_value = True
    response = client.delete("/api/sessions/1")
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "Session deleted"}

def test_delete_session_not_found(mock_chat_service):
    mock_chat_service.delete_session.return_value = False
    response = client.delete("/api/sessions/1")
    assert response.status_code == 404
