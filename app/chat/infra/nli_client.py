"""Natural Language Inference (NLI) infrastructure adapter."""

import httpx
import structlog
from typing import Any

from app.thesis.ram.interfaces import INLIModel, NLIResult

logger = structlog.get_logger(__name__)

LABEL_ENTAILMENT = "entailment"
LABEL_NEUTRAL = "neutral"
LABEL_CONTRADICTION = "contradiction"

_LABEL_MAP: dict[str, str] = {
    "entailment": LABEL_ENTAILMENT,
    "neutral": LABEL_NEUTRAL,
    "contradiction": LABEL_CONTRADICTION,
    "label_0": LABEL_ENTAILMENT,
    "label_1": LABEL_NEUTRAL,
    "label_2": LABEL_CONTRADICTION,
}

class NLIClient(INLIModel):
    """Infrastructure adapter for NLI inference via the Infinity HTTP server."""

    def __init__(self, base_url: str, model: str):
        self.model = model
        self._nli_sep = " </s></s> " if "roberta" in model.lower() else " [SEP] "
        
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info("chat.nli.initialized", model=model, base_url=base_url, sep=self._nli_sep)

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        max_premise_chars = 1500
        if len(premise) > max_premise_chars:
            premise = premise[:max_premise_chars]

        text = f"{premise}{self._nli_sep}{hypothesis}"
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
        except Exception as e:
            logger.warning("chat.nli.failed", error=str(e))
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
                neutral_score=0.5,
            )

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> NLIResult:
        items = data.get("data", [])
        if not items or not items[0]:
            return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0, neutral_score=0.5)

        predictions = items[0]
        
        if isinstance(predictions, list) and all(isinstance(p, dict) for p in predictions):
            score_dict = {str(p.get("label", "")): float(p.get("score", 0.0)) for p in predictions}
            return NLIClient._parse_raw_scores(score_dict)
        elif isinstance(predictions, dict):
            score_field = predictions.get("score")
            if isinstance(score_field, dict):
                return NLIClient._parse_raw_scores(score_field)
            return NLIClient._parse_top1(
                label=str(predictions.get("label", "")),
                score=float(score_field) if score_field is not None else 0.0,
            )

        return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0, neutral_score=0.5)

    @staticmethod
    def _parse_raw_scores(score_dict: dict[str, Any]) -> NLIResult:
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

        return NLIResult(
            label=best_label,
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            neutral_score=neutral_score,
        )

    @staticmethod
    def _parse_top1(label: str, score: float) -> NLIResult:
        canonical = _LABEL_MAP.get(label.lower(), LABEL_NEUTRAL)

        entailment_score = score if canonical == LABEL_ENTAILMENT else 0.0
        contradiction_score = score if canonical == LABEL_CONTRADICTION else 0.0
        neutral_score = score if canonical == LABEL_NEUTRAL else 0.0

        return NLIResult(
            label=canonical,
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            neutral_score=neutral_score,
        )

    async def close(self) -> None:
        await self._client.aclose()
