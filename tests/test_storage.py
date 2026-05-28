import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from src.infra import LocalStorageProvider


@pytest.fixture
def temp_upload_dir(tmp_path):
    return str(tmp_path)

@pytest.mark.asyncio
async def test_local_storage_init(temp_upload_dir):
    provider = LocalStorageProvider(temp_upload_dir)
    assert provider.upload_dir == temp_upload_dir
    assert os.path.exists(temp_upload_dir)

@pytest.mark.asyncio
async def test_save_file_success(temp_upload_dir):
    provider = LocalStorageProvider(temp_upload_dir)

    # Mock an UploadFile
    mock_file = AsyncMock()
    mock_file.read.side_effect = [b"test data chunk 1", b" chunk 2", b""]

    # Mock uuid to predict the filename
    test_uuid = uuid.uuid4()
    with patch("src.infra.storage.uuid.uuid4", return_value=test_uuid):
        final_path = await provider.save_file(mock_file, ".txt")

    assert final_path == os.path.join(temp_upload_dir, f"{test_uuid}.txt")
    assert os.path.exists(final_path)
    with open(final_path, "rb") as f:
        assert f.read() == b"test data chunk 1 chunk 2"

@pytest.mark.asyncio
async def test_save_file_exception_cleanup(temp_upload_dir):
    provider = LocalStorageProvider(temp_upload_dir)

    # Mock an UploadFile that raises an exception on read
    mock_file = AsyncMock()
    mock_file.read.side_effect = Exception("Read failed")

    test_uuid = uuid.uuid4()
    with patch("src.infra.storage.uuid.uuid4", return_value=test_uuid):
        with pytest.raises(Exception, match="Read failed"):
            await provider.save_file(mock_file, ".txt")

    # Temp file should be cleaned up
    temp_path = os.path.join(temp_upload_dir, f"{test_uuid}.txt.tmp")
    assert not os.path.exists(temp_path)

@pytest.mark.asyncio
async def test_delete_file_success(temp_upload_dir):
    provider = LocalStorageProvider(temp_upload_dir)

    file_path = os.path.join(temp_upload_dir, "test_file.txt")
    with open(file_path, "w") as f:
        f.write("content")

    result = await provider.delete_file(file_path)
    assert result is True
    assert not os.path.exists(file_path)

@pytest.mark.asyncio
async def test_delete_file_not_found(temp_upload_dir):
    provider = LocalStorageProvider(temp_upload_dir)

    file_path = os.path.join(temp_upload_dir, "nonexistent.txt")
    result = await provider.delete_file(file_path)
    assert result is False
