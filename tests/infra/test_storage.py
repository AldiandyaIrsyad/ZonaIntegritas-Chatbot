"""Tests for storage infrastructure module."""

import os
import uuid
import tempfile
import pytest

from unittest.mock import AsyncMock

from app.infra.storage import LocalStorageProvider

@pytest.fixture
def upload_dir():
    """Fixture providing a temporary directory for uploads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def storage(upload_dir):
    """Fixture providing a LocalStorageProvider instance."""
    return LocalStorageProvider(upload_dir=upload_dir)

@pytest.fixture
def mock_upload_file():
    """Fixture providing a mock UploadFile."""
    mock_file = AsyncMock()
    # Mock read to return some data once, then empty bytes to simulate EOF
    mock_file.read.side_effect = [b"test file content", b""]
    return mock_file

async def test_save_file_success(storage, mock_upload_file, upload_dir):
    """Test atomic file saving."""
    file_extension = ".txt"
    final_path = await storage.save_file(mock_upload_file, file_extension)
    
    assert final_path.startswith(upload_dir)
    assert final_path.endswith(file_extension)
    assert os.path.exists(final_path)
    
    with open(final_path, "rb") as f:
        assert f.read() == b"test file content"
        
    # Temporary file should be gone
    assert not os.path.exists(final_path + ".tmp")

async def test_save_file_exception_cleanup(storage, upload_dir):
    """Test staging file cleanup on failure."""
    # Mock an upload file that raises an exception during read
    mock_file = AsyncMock()
    mock_file.read.side_effect = Exception("Read failed")
    
    with pytest.raises(Exception, match="Read failed"):
        await storage.save_file(mock_file, ".txt")
        
    # Check that no temp files or final files are left in the directory
    files_in_dir = os.listdir(upload_dir)
    assert len(files_in_dir) == 0

async def test_delete_file_success(storage, upload_dir):
    """Test deleting an existing file."""
    # Create a test file
    test_path = os.path.join(upload_dir, "test.txt")
    with open(test_path, "wb") as f:
        f.write(b"content")
        
    assert os.path.exists(test_path)
    
    result = await storage.delete_file(test_path)
    
    assert result is True
    assert not os.path.exists(test_path)

async def test_delete_file_not_found(storage, upload_dir):
    """Test deleting a non-existent file."""
    test_path = os.path.join(upload_dir, "does_not_exist.txt")
    
    result = await storage.delete_file(test_path)
    
    assert result is False

async def test_delete_file_empty_path(storage):
    """Test deleting with an empty path."""
    result = await storage.delete_file("")
    assert result is False

async def test_delete_file_exception(storage, upload_dir, mocker):
    """Test handling exception during deletion."""
    test_path = os.path.join(upload_dir, "test.txt")
    with open(test_path, "wb") as f:
        f.write(b"content")
        
    mocker.patch("os.remove", side_effect=PermissionError("Permission denied"))
    
    result = await storage.delete_file(test_path)
    
    assert result is False
