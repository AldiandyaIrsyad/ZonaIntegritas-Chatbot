"""
Prompt Injection detection infrastructure adapter.

Wraps HTTP calls to the Infinity `/classify` endpoint, replacing the previous
in-process HuggingFace transformers pipeline. Concurrency is now handled
entirely by the Infinity server — multiple in-flight classify requests are
processed in parallel without any GIL or thread-pool contention.

Model: ProtectAI/deberta-v3-base-prompt-injection-v2
Labels: SAFE (0) | INJECTION (1) | JAILBREAK (2)
"""
import logging
from typing import Tuple

import httpx

logger = logging.getLogger(__name__)


class PromptGuardProvider:
    """
    Infrastructure adapter for Prompt Injection detection via Infinity HTTP.

    Designed as a singleton: instantiate once at startup (via lru_cache in
    ivm/dependency.py), reuse across all requests. `check_prompt()` is a
    direct async method — no thread offload required.
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

        Returns:
            (True, "Safe")                          — benign input
            (False, "Policy violation: <detail>")   — injection / jailbreak detected
            (False, "Service unavailable")          — Infinity unreachable / error
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
            if not items or not items[0]:
                logger.warning("Empty classify response from Infinity (PromptGuard)")
                return False, "Service unavailable"

            predictions = items[0]
            if isinstance(predictions, dict):
                predictions = [predictions]

            for pred in predictions:
                label = str(pred.get("label", "")).upper()
                score = float(pred.get("score", 0.0))

                # Labels from ProtectAI/deberta-v3-base-prompt-injection-v2:
                # SAFE (benign), INJECTION, JAILBREAK — or generic LABEL_1/LABEL_2
                if (
                    "INJECTION" in label
                    or "JAILBREAK" in label
                    or "LABEL_1" in label
                    or "LABEL_2" in label
                ):
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
