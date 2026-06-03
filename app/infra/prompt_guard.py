"""
Prompt Injection detection infrastructure adapter.

Wraps HTTP calls to the Infinity `/classify` endpoint, replacing the previous
in-process HuggingFace transformers pipeline. Concurrency is now handled
entirely by the Infinity server — multiple in-flight classify requests are
processed in parallel without any GIL or thread-pool contention.

Model: meta-llama/Llama-Prompt-Guard-2-86M
Labels: BENIGN (safe input) | MALICIOUS (prompt injection or jailbreak attempt)
"""
import httpx
import structlog

from app.core.interfaces.ai import PromptGuardResult

logger = structlog.get_logger(__name__)

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
            "PromptGuardProvider initialised",
            model=model,
            base_url=base_url,
            threshold=security_threshold,
        )

    async def check_prompt(self, text: str) -> PromptGuardResult:
        """
        Check text for prompt injection via Infinity classify.

        Args:
            text (str): The user's input prompt to check.

        Returns:
            PromptGuardResult: Result of the injection check.
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
                return PromptGuardResult(is_safe=False, message="Service unavailable")

            predictions = items[0]
            
            if isinstance(predictions, dict) and "results" in predictions:
                predictions = predictions["results"]
            elif isinstance(predictions, dict):
                predictions = [predictions]

            if not predictions:
                logger.warning("No valid predictions found in Infinity response.")
                return PromptGuardResult(is_safe=False, message="Service unavailable")

            for pred in predictions:
                label = str(pred.get("label", "")).upper()
                score = float(pred.get("score", 0.0))

                if label == "LABEL_0":
                    label = "BENIGN"
                elif label == "LABEL_1":
                    label = "MALICIOUS"

                logger.debug(
                    "PromptGuard prediction",
                    label=label,
                    score=score,
                    threshold=self.security_threshold,
                )
                
                if "BENIGN" not in label:
                    if score >= self.security_threshold:
                        return PromptGuardResult(
                            is_safe=False,
                            message=f"Policy violation: {label} (Score: {score:.2f} >= {self.security_threshold})",
                        )

            return PromptGuardResult(is_safe=True, message="Safe")

        except httpx.HTTPStatusError as e:
            logger.error(
                "Infinity classify returned HTTP error for PromptGuard",
                status_code=e.response.status_code,
                error=str(e),
            )
            return PromptGuardResult(is_safe=False, message="Service unavailable")
        except Exception as e:
            logger.error("PromptGuard classify request failed", error=str(e))
            return PromptGuardResult(is_safe=False, message="Service unavailable")

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()