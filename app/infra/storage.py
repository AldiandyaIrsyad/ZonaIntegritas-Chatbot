"""Local filesystem storage adapter.

Provides :class:`LocalStorageProvider`, a concrete implementation of the
:class:`~app.core.interfaces.infra.IStorageProvider` Protocol for persisting
uploaded files to disk.

File writes use an atomic temp-file + ``os.replace`` pattern to ensure that
readers never observe a partially written file.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import aiofiles
import structlog
from fastapi import UploadFile

from app.core.interfaces.infra import IStorageProvider

logger = structlog.get_logger(__name__)


class LocalStorageProvider:
    """Filesystem implementation of :class:`~app.core.interfaces.infra.IStorageProvider`.

    Writes uploaded files atomically: data is buffered to a ``.tmp`` staging
    file, then renamed to the final destination via ``os.replace``.  This
    prevents readers from observing partial content under concurrent load.

    On write failure, the staging file is cleaned up before re-raising.

    Satisfies the :class:`~app.core.interfaces.infra.IStorageProvider`
    Protocol structurally.

    Args:
        upload_dir: Directory where uploaded files are stored.  Created
                    automatically if it does not already exist.
    """

    def __init__(self, upload_dir: str) -> None:
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        logger.info("LocalStorageProvider initialised", upload_dir=upload_dir)

    async def save_file(self, file: UploadFile, file_extension: str) -> str:
        """Persist an uploaded file to local storage atomically.

        Streams the upload in 1 MiB chunks to a ``<uuid>.tmp`` staging file,
        then renames it to ``<uuid><ext>`` on success.

        Args:
            file: The FastAPI :class:`~fastapi.UploadFile` object to persist.
            file_extension: File extension including the leading dot
                            (e.g. ``".pdf"``).

        Returns:
            Absolute path to the stored file.

        Raises:
            Exception: Re-raised after cleaning up the staging file on any
                I/O error.
        """
        unique_name = f"{uuid.uuid4()}{file_extension}"
        final_path = os.path.join(self.upload_dir, unique_name)
        temp_path = final_path + ".tmp"

        try:
            async with aiofiles.open(temp_path, "wb") as out_file:
                while chunk := await file.read(1024 * 1024):
                    await out_file.write(chunk)

            await asyncio.to_thread(os.replace, temp_path, final_path)

            logger.debug(
                "storage.save.complete",
                filename=unique_name,
                path=final_path,
            )
            return final_path

        except Exception as exc:
            logger.error(
                "storage.save.failed",
                temp_path=temp_path,
                final_path=final_path,
                error=str(exc),
            )
            # Best-effort cleanup of the incomplete staging file
            try:
                if os.path.exists(temp_path):
                    await asyncio.to_thread(os.remove, temp_path)
            except Exception as cleanup_exc:
                logger.warning(
                    "storage.save.cleanup_failed",
                    temp_path=temp_path,
                    error=str(cleanup_exc),
                )
            raise

    async def delete_file(self, file_path: str) -> bool:
        """Remove a file from local storage.

        Args:
            file_path: Absolute path of the file to delete.

        Returns:
            ``True`` if the file was removed.
            ``False`` if ``file_path`` was empty or the file was not found.
        """
        if not file_path:
            return False

        try:
            await asyncio.to_thread(os.remove, file_path)
            logger.debug("storage.delete.complete", file_path=file_path)
            return True
        except FileNotFoundError:
            logger.warning("storage.delete.not_found", file_path=file_path)
            return False
        except Exception as exc:
            logger.error(
                "storage.delete.failed",
                file_path=file_path,
                error=str(exc),
            )
            return False
