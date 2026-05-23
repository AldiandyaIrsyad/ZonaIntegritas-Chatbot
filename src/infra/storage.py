import os
import aiofiles
import uuid
import asyncio
from abc import ABC, abstractmethod
from typing import BinaryIO
from fastapi import UploadFile
from src.core.config import get_storage_settings

class StorageProvider(ABC):
    """Abstract base class for storage providers."""

    @abstractmethod
    async def save_file(self, file: UploadFile, file_extension: str) -> str:
        """Saves a file and returns its path/URI."""
        pass

    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Deletes a file given its path/URI. Returns True if deleted."""
        pass


class LocalStorageProvider(StorageProvider):
    """Local file system implementation for storage."""

    def __init__(self):
        settings = get_storage_settings()
        self.upload_dir = settings.upload_dir
        # Ensure the directory exists
        os.makedirs(self.upload_dir, exist_ok=True)

    async def save_file(self, file: UploadFile, file_extension: str) -> str:
        """
        Saves the file to local storage.
        Uses a temporary file and an atomic rename to prevent race conditions or partial writes.
        """
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        final_path = os.path.join(self.upload_dir, unique_filename)
        temp_path = final_path + ".tmp"

        try:
            async with aiofiles.open(temp_path, 'wb') as out_file:
                while chunk := await file.read(1024 * 1024):
                    await out_file.write(chunk)
            
            os.replace(temp_path, final_path)
            
            return final_path
            
        except Exception as e:
            # Clean up the temp file if something went wrong during write
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    async def delete_file(self, file_path: str) -> bool:
        """Deletes the file from local storage."""
        if not file_path:
            return False
            
        try:
            await asyncio.to_thread(os.remove, file_path)
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            # Pass

            return False
