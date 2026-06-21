"""Research-oriented observability utilities for storing large payloads and logs."""

import json
import os
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ResearchLogger:
    """Handles persistence of large raw outputs for research purposes."""

    def __init__(self, base_log_dir: str = "logs/raw_outputs") -> None:
        """Initialize the research logger.

        Args:
            base_log_dir: The root directory where raw outputs should be stored.
        """
        self.base_log_dir = os.path.abspath(base_log_dir)
        os.makedirs(self.base_log_dir, exist_ok=True)

    def save_raw_output(
        self, component: str, document_name: str, doc_hash: str, payload: dict[str, Any] | list[Any]
    ) -> str:
        """Save a large JSON payload to disk and return the path.

        Args:
            component: The name of the component generating the log (e.g. 'document_parser').
            document_name: The name of the document being processed.
            doc_hash: The SHA256 hash of the document for provenance.
            payload: The raw JSON output to save.

        Returns:
            The absolute path to the saved JSON file.
        """
        component_dir = os.path.join(self.base_log_dir, component)
        os.makedirs(component_dir, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_doc_name = os.path.basename(document_name).replace(" ", "_")
        filename = f"{timestamp}_{doc_hash[:8]}_{safe_doc_name}.json"

        file_path = os.path.join(component_dir, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.debug("research_logger.saved_raw_output", file_path=file_path)
            return file_path
        except Exception as exc:
            logger.error("research_logger.save_failed", error=str(exc), file_path=file_path)
            raise
