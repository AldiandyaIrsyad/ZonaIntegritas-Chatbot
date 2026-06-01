"""
Prompt Injection detection infrastructure adapter.

Wraps HTTP calls to the Infinity `/classify` endpoint, replacing the previous
in-process HuggingFace transformers pipeline. Concurrency is now handled
entirely by the Infinity server — multiple in-flight classify requests are
processed in parallel without any GIL or thread-pool contention.

Model: meta-llama/Llama-Prompt-Guard-2-86M
Labels: BENIGN (safe input) | MALICIOUS (prompt injection or jailbreak attempt)
"""
import logging
from typing import Tuple

import httpx

logger = logging.getLogger(__name__)

class PromptGuardProvider:
    """
    Infrastructure adapter for Prompt Injection detection via Infinity HTTP.

    Designed as a singleton: instantiated lazily on the first request (via lru_cache in
    ivm/dependency.py), and reused across all subsequent requests. `check_prompt()` is a
    direct async method — no thread offload required.

    Args:
        base_url (str): The base URL of the Infinity server.
        model (str): The prompt guard model to use.
        security_threshold (float, optional): Score threshold to trigger a violation. Defaults to 0.75.
    """

    def __init__(self, base_url: str, model: str, security_threshold: float = 0.75):
        self.model = model
        self.security_threshold = security_threshold
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info(
            "PromptGuardProvider initialised — model=%s, base_url=%s, threshold=%.2f",
            model, base_url, security_threshold,
        )

    async def check_prompt(self, text: str) -> Tuple[bool, str]:
        """
        Check text for prompt injection via Infinity classify.

        Args:
            text (str): The user's input prompt to check.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - `True`, "Safe"                          — benign input
                - `False`, "Policy violation: <detail>"   — injection / jailbreak detected
                - `False`, "Service unavailable"          — Infinity unreachable / error
        """
        try:
            response = await self._client.post(
                "/classify",
                json={
                    "model": self.model,
                    "input": [text],
                },
            )
            response.raise_for_status()
            data = response.json()

            items = data.get("data", [])
            if not items:
                logger.warning("Empty classify response from Infinity (PromptGuard)")
                return False, "Service unavailable"

            predictions = items[0]
            
            if isinstance(predictions, dict) and "results" in predictions:
                predictions = predictions["results"]
            elif isinstance(predictions, dict):
                predictions = [predictions]

            if not predictions:
                logger.warning("No valid predictions found in Infinity response.")
                return False, "Service unavailable"

            for pred in predictions:
                label = str(pred.get("label", "")).upper()
                score = float(pred.get("score", 0.0))

                if label == "LABEL_0":
                    label = "BENIGN"
                elif label == "LABEL_1":
                    label = "MALICIOUS"

                logger.debug(
                    "PromptGuard prediction — label=%s, score=%.4f, threshold=%.2f",
                    label, score, self.security_threshold,
                )
                
                if "BENIGN" not in label:
                    if score >= self.security_threshold:
                        return (
                            False,
                            f"Policy violation: {label} (Score: {score:.2f} >= {self.security_threshold})",
                        )

            return True, "Safe"

        except httpx.HTTPStatusError as e:
            logger.error(
                "Infinity classify returned HTTP %d for PromptGuard: %s",
                e.response.status_code, e,
            )
            return False, "Service unavailable"
        except Exception as e:
            logger.error("PromptGuard classify request failed: %s", e)
            return False, "Service unavailable"

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()