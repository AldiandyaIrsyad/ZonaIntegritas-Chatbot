"""
Natural Language Inference (NLI) infrastructure adapter.

Wraps a Hugging Face transformers text-classification pipeline for NLI.
The model is loaded once at startup (singleton via lru_cache in dependency.py)
and inference runs in a thread pool via anyio to avoid blocking FastAPI.

Supported models (configured via RAM_NLI_MODEL env var):
- StevenLimcorn/indo-roberta-indonli  (default)
- LazarusNLP/indobert-lite-base-p1-indonli-distil-mdeberta
"""
import logging
from dataclasses import dataclass
from typing import List

import anyio
from transformers import pipeline

logger = logging.getLogger(__name__)

# Canonical label names — model-specific label strings are mapped to these.
LABEL_ENTAILMENT = "entailment"
LABEL_NEUTRAL = "neutral"
LABEL_CONTRADICTION = "contradiction"

# Known label mappings for supported models.
# Keys are lowercased label strings as returned by the pipeline.
_LABEL_MAP: dict[str, str] = {
    # StevenLimcorn/indo-roberta-indonli
    "entailment": LABEL_ENTAILMENT,
    "neutral": LABEL_NEUTRAL,
    "contradiction": LABEL_CONTRADICTION,
    # Generic LABEL_N fallbacks (model may output these before id2label resolution)
    "label_0": LABEL_ENTAILMENT,
    "label_1": LABEL_NEUTRAL,
    "label_2": LABEL_CONTRADICTION,
}


@dataclass
class NLIResult:
    """Result from a single NLI inference call."""
    label: str              # "entailment" | "neutral" | "contradiction"
    entailment_score: float  # confidence that hypothesis is entailed by premise (0.0–1.0)
    contradiction_score: float  # confidence that hypothesis contradicts premise (0.0–1.0)


class NLIProvider:
    """
    Infrastructure adapter for local NLI inference using transformers.

    Designed as a singleton: instantiate once at startup, reuse across
    all requests. Inference is offloaded to a thread so the async event
    loop is never blocked.
    """

    def __init__(self, model: str, device: int = -1, max_length: int = 512):
        logger.info(
            "Initializing NLI pipeline with model='%s', device=%d", model, device
        )
        self.max_length = max_length
        # top_k=None returns scores for all labels — required for multi-label scoring.
        self._pipeline = pipeline(
            "text-classification",
            model=model,
            device=device,
            top_k=None,
        )
        logger.info("NLI pipeline ready.")

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI asynchronously.

        Args:
            premise: The reference context (KB parent chunk texts).
            hypothesis: The text to verify against the premise (LLM sentence).

        Returns:
            NLIResult with normalized label and per-class scores.
        """
        try:
            raw_results: List[dict] = await anyio.to_thread.run_sync(
                lambda: self._run_pipeline(premise, hypothesis)
            )
            return self._parse_results(raw_results)
        except Exception:
            logger.warning(
                "NLI inference failed, returning neutral default", exc_info=True
            )
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
            )

    def _run_pipeline(self, premise: str, hypothesis: str) -> List[dict]:
        """Synchronous pipeline call — runs in a thread pool."""
        # For NLI models, `text` is premise and `text_pair` is hypothesis.
        # top_k=None means the pipeline returns ALL label scores.
        return self._pipeline(  # type: ignore[operator]
            {"text": premise, "text_pair": hypothesis},
            truncation=True,
            max_length=self.max_length,
        )

    def _parse_results(self, raw_results: List[dict]) -> NLIResult:
        """Normalize raw pipeline output into NLIResult.

        Handles both named labels ('entailment') and generic labels
        ('LABEL_0') so the adapter works with any compatible NLI model.
        """
        scores: dict[str, float] = {}
        for item in raw_results:
            raw_label = str(item.get("label", "")).lower()
            canonical = _LABEL_MAP.get(raw_label, LABEL_NEUTRAL)
            scores[canonical] = float(item.get("score", 0.0))

        entailment_score = scores.get(LABEL_ENTAILMENT, 0.0)
        neutral_score = scores.get(LABEL_NEUTRAL, 0.0)
        contradiction_score = scores.get(LABEL_CONTRADICTION, 0.0)

        # Pick the label with the highest score
        best_label = max(
            [(LABEL_ENTAILMENT, entailment_score),
             (LABEL_NEUTRAL, neutral_score),
             (LABEL_CONTRADICTION, contradiction_score)],
            key=lambda x: x[1],
        )[0]

        logger.debug(
            "NLI result: label=%s, entailment=%.3f, contradiction=%.3f",
            best_label,
            entailment_score,
            contradiction_score,
        )

        return NLIResult(
            label=best_label,
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
        )

    async def close(self) -> None:
        """No-op — transformers pipelines don't hold external connections."""
        pass
