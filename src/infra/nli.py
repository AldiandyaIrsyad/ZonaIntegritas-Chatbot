"""
Natural Language Inference (NLI) infrastructure adapter.

Wraps HTTP calls to the Infinity `/classify` endpoint instead of loading
a local HuggingFace pipeline. This gives true async concurrency — multiple
NLI calls in flight simultaneously — without any GIL/thread-pool contention.

The Infinity server is responsible for batching, GPU scheduling, and model
lifecycle. This client is a thin, stateless HTTP wrapper.

Supported models (configured via INFINITY_NLI_MODEL env var):
- StevenLimcorn/indo-roberta-indonli  (default)
- LazarusNLP/indobert-lite-base-p1-indonli-distil-mdeberta

Infinity classify endpoint reference:
  POST /classify
  Body: {"model": "...", "input": ["text"], "raw_scores": true}
  Response (raw_scores=true): scores dict per label
  Response (raw_scores=false / fallback): single top-1 {label, score}
"""
import logging
from dataclasses import dataclass
from typing import List

import httpx

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

# Separator used to join premise + hypothesis into a single string.
# HuggingFace NLI pipelines use [SEP] internally; we replicate that here.
# _NLI_SEP = " [SEP] " # BERT
_NLI_SEP = " </s></s> " # RoBERTa



@dataclass
class NLIResult:
    """Result from a single NLI inference call."""
    label: str                          # "entailment" | "neutral" | "contradiction"
    entailment_score: float     = 0.0   # confidence that hypothesis is entailed by premise (0.0–1.0)
    contradiction_score: float  = 0.0   # confidence that hypothesis contradicts premise (0.0–1.0)
    neutral_score: float        = 0.0   # confidence that hypothesis is neutral to premise (0.0–1.0)


class NLIProvider:
    """
    Infrastructure adapter for NLI inference via the Infinity HTTP server.

    Designed as a singleton: instantiate once at startup (via lru_cache in
    dependency.py), reuse across all requests. Each `check()` call is a
    true async HTTP request — no threads, no GIL contention — allowing
    multiple NLI tasks to be in-flight simultaneously during LLM streaming.

    Infinity classify endpoint behaviour:
    - With `raw_scores=True`: response contains a score dict with all labels.
    - Without (or unsupported): response contains single top-1 {label, score}.
    Both shapes are handled gracefully.
    """

    def __init__(self, base_url: str, model: str):
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info("NLIProvider initialised — model=%s, base_url=%s", model, base_url)

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI asynchronously via the Infinity classify endpoint.

        Args:
            premise: The reference context (KB parent chunk texts).
            hypothesis: The text to verify against the premise (LLM sentence).

        Returns:
            NLIResult with normalized label and per-class scores.
        """
        # NLI models expect (premise, hypothesis) as a text-pair.
        # Infinity classify takes a single string; we join with [SEP].
        text = f"{premise}{_NLI_SEP}{hypothesis}"
        try:
            response = await self._client.post(
                "/classify",
                json={
                    "model": self.model,
                    "input": [text],
                    "raw_scores": True,
                },
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)
        except Exception:
            logger.warning(
                "NLI inference failed, returning neutral default", exc_info=True
            )
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
                neutral_score=0.5,
            )

    def _parse_response(self, data: dict) -> NLIResult:
        """Parse Infinity classify response into NLIResult.

        Handles two shapes:
        1. raw_scores=True  → data["data"][0]["score"] is a dict of {label: score}
        2. Fallback         → data["data"][0]["score"] is a float, "label" is a str
        """
        items = data.get("data", [])
        if not items or not items[0]:
            logger.warning("Empty classify response from Infinity")
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
                neutral_score=0.5,
            )

        predictions = items[0]
        
        if isinstance(predictions, list) and all(isinstance(p, dict) for p in predictions):
            # List of dicts: [{"label": "Neutral", "score": 0.66}, ...]
            score_dict = {str(p.get("label", "")): float(p.get("score", 0.0)) for p in predictions}
            return self._parse_raw_scores(score_dict)
        elif isinstance(predictions, dict):
            # Fallback if it returns a single dict instead of list of dicts
            score_field = predictions.get("score")
            if isinstance(score_field, dict):
                return self._parse_raw_scores(score_field)
            return self._parse_top1(
                label=str(predictions.get("label", "")),
                score=float(score_field) if score_field is not None else 0.0,
            )

        logger.warning("Unexpected classify response shape from Infinity")
        return NLIResult(
            label=LABEL_NEUTRAL,
            entailment_score=0.5,
            contradiction_score=0.0,
            neutral_score=0.5,
        )

    def _parse_raw_scores(self, score_dict: dict) -> NLIResult:
        """Parse a full label→score mapping into NLIResult."""
        scores: dict[str, float] = {}
        for raw_label, score in score_dict.items():
            canonical = _LABEL_MAP.get(raw_label.lower(), LABEL_NEUTRAL)
            scores[canonical] = float(score)

        entailment_score = scores.get(LABEL_ENTAILMENT, 0.0)
        neutral_score = scores.get(LABEL_NEUTRAL, 0.0)
        contradiction_score = scores.get(LABEL_CONTRADICTION, 0.0)

        best_label = max(
            [
                (LABEL_ENTAILMENT, entailment_score),
                (LABEL_NEUTRAL, neutral_score),
                (LABEL_CONTRADICTION, contradiction_score),
            ],
            key=lambda x: x[1],
        )[0]

        logger.debug(
            "NLI result (raw_scores): label=%s, entailment=%.3f, contradiction=%.3f",
            best_label, entailment_score, contradiction_score,
        )
        return NLIResult(
            label=best_label,
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            neutral_score=neutral_score,
        )

    def _parse_top1(self, label: str, score: float) -> NLIResult:
        """Parse a single top-1 label+score into NLIResult (fallback path)."""
        canonical = _LABEL_MAP.get(label.lower(), LABEL_NEUTRAL)

        entailment_score = score if canonical == LABEL_ENTAILMENT else 0.0
        contradiction_score = score if canonical == LABEL_CONTRADICTION else 0.0

        neutral_score = score if canonical == LABEL_NEUTRAL else 0.0

        logger.debug(
            "NLI result (top-1 fallback): label=%s, score=%.3f", canonical, score
        )
        return NLIResult(
            label=canonical,
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            neutral_score=neutral_score,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
