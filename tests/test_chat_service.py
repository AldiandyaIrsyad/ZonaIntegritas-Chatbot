from unittest.mock import AsyncMock, MagicMock

import pytest

from src.chat import ChatService


@pytest.fixture
def mock_repository():
    return AsyncMock()

@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
def mock_retrieval_service():
    service = AsyncMock()
    # Default: no RAG context retrieved
    service.retrieve_context.return_value = []
    return service

@pytest.fixture
def mock_storage():
    return AsyncMock()

@pytest.fixture
def mock_document_parser():
    return AsyncMock()

@pytest.fixture
def mock_reranker():
    return AsyncMock()

@pytest.fixture
def mock_vector_store():
    return AsyncMock()

@pytest.fixture
def mock_embedding_provider():
    return AsyncMock()

@pytest.fixture
def chat_service(
    mock_repository,
    mock_llm,
    mock_retrieval_service,
    mock_storage,
    mock_document_parser,
    mock_reranker,
    mock_vector_store,
    mock_embedding_provider
):
    return ChatService(
        repository=mock_repository,
        llm_service=mock_llm,
        retrieval_service=mock_retrieval_service,
        storage=mock_storage,
        document_parser=mock_document_parser,
        reranker=mock_reranker,
        vector_store=mock_vector_store,
        embedding_provider=mock_embedding_provider
    )

@pytest.mark.asyncio
async def test_list_sessions(chat_service, mock_repository):
    mock_session = MagicMock()
    mock_session.id = "123"
    mock_session.title = "Test Session"
    mock_repository.get_all_sessions.return_value = [mock_session]

    result = await chat_service.list_sessions()
    assert len(result) == 1
    assert result[0] == {"id": "123", "title": "Test Session"}
    mock_repository.get_all_sessions.assert_called_once()

@pytest.mark.asyncio
async def test_create_new_session(chat_service, mock_repository):
    mock_session = MagicMock()
    mock_session.id = "new_id"
    mock_session.title = "New Chat"
    mock_repository.create_session.return_value = mock_session

    result = await chat_service.create_new_session()
    assert result == {"id": "new_id", "title": "New Chat"}
    mock_repository.create_session.assert_called_once()

@pytest.mark.asyncio
async def test_get_session_details(chat_service, mock_repository):
    mock_session = MagicMock()
    mock_session.title = "Test Title"
    mock_msg = MagicMock()
    mock_msg.role = "user"
    mock_msg.content = "Hello"
    mock_session.messages = [mock_msg]
    mock_session.documents = []
    mock_repository.get_session_by_id.return_value = mock_session

    result = await chat_service.get_session_details("123")
    assert result == {
        "title": "Test Title",
        "messages": [{"role": "user", "content": "Hello"}],
        "documents": []
    }

@pytest.mark.asyncio
async def test_get_session_details_not_found(chat_service, mock_repository):
    mock_repository.get_session_by_id.return_value = None
    result = await chat_service.get_session_details("invalid_id")
    assert result is None

@pytest.mark.asyncio
async def test_process_chat_message(chat_service, mock_repository, mock_llm):
    # Setup mock session
    mock_session = MagicMock()
    mock_session.title = "New Chat"
    mock_session.messages = []
    mock_repository.get_session_by_id.return_value = mock_session
    
    # Mock LLM stream
    async def mock_stream(*args, **kwargs):
        yield "Hello "
        yield "World"
    
    mock_llm.stream_response.return_value = mock_stream()

    response = await chat_service.process_chat_message("123", "Test message")
    
    # Process the stream
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert "".join(chunks) == "Hello World"
    
    # Verify repository calls
    mock_repository.update_session_title.assert_called_once_with(mock_session, "Test message")
    mock_repository.create_message.assert_any_call("123", "user", "Test message")
    mock_repository.create_message.assert_any_call("123", "assistant", "Hello World")

@pytest.mark.asyncio
async def test_delete_session(chat_service, mock_repository):
    mock_repository.delete_session.return_value = True
    result = await chat_service.delete_session("123")
    assert result is True
    mock_repository.delete_session.assert_called_once_with("123")
