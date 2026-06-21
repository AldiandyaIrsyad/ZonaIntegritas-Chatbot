"""Prompt Injection detection infrastructure adapter."""

import httpx
import structlog
from typing import List, Dict, Any

from app.thesis.ivm.interfaces import ISafetyModel, SafetyResult

logger = structlog.get_logger(__name__)

class PromptGuardClient(ISafetyModel):
    """Infrastructure adapter for Prompt Injection detection via Infinity HTTP."""

    def __init__(self, base_url: str, model: str, security_threshold: float = 0.75):
        self.model = model
        self.security_threshold = security_threshold
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info(
            "chat.prompt_guard.initialized",
            model=model,
            base_url=base_url,
            threshold=security_threshold,
        )

    async def check_prompt(self, text: str) -> SafetyResult:
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
                return SafetyResult(is_safe=False, message="Service unavailable")

            predictions = items[0]
            if isinstance(predictions, dict) and "results" in predictions:
                predictions = predictions["results"]
            elif isinstance(predictions, dict):
                predictions = [predictions]

            if not predictions:
                logger.warning("No valid predictions found in Infinity response.")
                return SafetyResult(is_safe=False, message="Service unavailable")

            for pred in predictions:
                label = str(pred.get("label", "")).upper()
                score = float(pred.get("score", 0.0))

                if label == "LABEL_0":
                    label = "BENIGN"
                elif label == "LABEL_1":
                    label = "MALICIOUS"

                logger.debug("chat.prompt_guard.prediction", label=label, score=score)
                
                if "BENIGN" not in label:
                    if score >= self.security_threshold:
                        return SafetyResult(
                            is_safe=False,
                            message=f"Policy violation: {label} (Score: {score:.2f} >= {self.security_threshold})",
                        )

            return SafetyResult(is_safe=True, message="Safe")

        except Exception as e:
            logger.error("chat.prompt_guard.failed", error=str(e))
            return SafetyResult(is_safe=False, message="Service unavailable")

    async def close(self) -> None:
        await self._client.aclose()
