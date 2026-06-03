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
import httpx
import structlog

from app.core.interfaces.ai import NLIResult

logger = structlog.get_logger(__name__)

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

# Separator is determined dynamically in __init__ based on the model name.



class NLIProvider:
    """
    Infrastructure adapter for NLI inference via the Infinity HTTP server.

    Designed as a singleton: instantiated lazily on the first request (via lru_cache in
    dependency.py), and reused across all subsequent requests. Each `check()` call is a
    true async HTTP request — no threads, no GIL contention — allowing
    multiple NLI tasks to be in-flight simultaneously during LLM streaming.

    Infinity classify endpoint behaviour:
    - With `raw_scores=True`: response contains a list of dicts with all labels and scores.
    - Without (or unsupported): response contains single top-1 {label, score}.
    Both shapes are handled gracefully.

    Args:
        base_url (str): The base URL of the Infinity server.
        model (str): The NLI model ID to use.
    """

    def __init__(self, base_url: str, model: str):
        self.model = model
        # Determine the correct separator based on the model architecture
        self._nli_sep = " </s></s> " if "roberta" in model.lower() else " [SEP] "
        
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info("NLIProvider initialised", model=model, base_url=base_url, sep=self._nli_sep)

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI asynchronously via the Infinity classify endpoint.

        Args:
            premise (str): The reference context (KB parent chunk texts).
            hypothesis (str): The text to verify against the premise (LLM sentence).

        Returns:
            NLIResult: NLIResult with normalized label and per-class scores.
        """
        # NLI models expect (premise, hypothesis) as a text-pair.
        # Infinity classify takes a single string; we join with the model-specific separator.
        
        # Truncate premise to avoid RoBERTa 512 token limit (approx 1500 chars).
        # We leave enough room for the hypothesis sentence.
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
            logger.warning(
                "NLI inference failed, returning neutral default", error=str(e), exc_info=True
            )
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
                neutral_score=0.5,
            )

    @staticmethod
    def _parse_response(data: dict) -> NLIResult:
        """Parse Infinity classify response into NLIResult.

        Handles two shapes:
        1. raw_scores=True  → data["data"][0]["score"] is a dict of {label: score}
        2. Fallback         → data["data"][0]["score"] is a float, "label" is a str

        Args:
            data (dict): The JSON response payload from Infinity.

        Returns:
            NLIResult: The parsed NLI result.
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
            return NLIProvider._parse_raw_scores(score_dict)
        elif isinstance(predictions, dict):
            # Fallback if it returns a single dict instead of list of dicts
            score_field = predictions.get("score")
            if isinstance(score_field, dict):
                return NLIProvider._parse_raw_scores(score_field)
            return NLIProvider._parse_top1(
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

    @staticmethod
    def _parse_raw_scores(score_dict: dict) -> NLIResult:
        """Parse a full label→score mapping into NLIResult.

        Args:
            score_dict (dict): A dictionary mapping raw labels to scores.

        Returns:
            NLIResult: The parsed NLI result.
        """
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
            "NLI result (raw_scores)",
            label=best_label,
            entailment=entailment_score,
            contradiction=contradiction_score,
        )
        return NLIResult(
            label=best_label,
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            neutral_score=neutral_score,
        )

    @staticmethod
    def _parse_top1(label: str, score: float) -> NLIResult:
        """Parse a single top-1 label+score into NLIResult (fallback path).

        Args:
            label (str): The predicted label.
            score (float): The confidence score.

        Returns:
            NLIResult: The parsed NLI result.
        """
        canonical = _LABEL_MAP.get(label.lower(), LABEL_NEUTRAL)

        entailment_score = score if canonical == LABEL_ENTAILMENT else 0.0
        contradiction_score = score if canonical == LABEL_CONTRADICTION else 0.0

        neutral_score = score if canonical == LABEL_NEUTRAL else 0.0

        logger.debug(
            "NLI result (top-1 fallback)", label=canonical, score=score
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
