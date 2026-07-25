"""Evaluator panel for dataset generation.

5-model majority voting panel. Each member independently evaluates a draft
item at temperature 0.0. An item is accepted if ≥4/5 members agree.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
import structlog

from app.shared.retry import external_api_retry
from app.thesis._eval._dataset_gen.config import DatasetGenSettings

logger = structlog.get_logger(__name__)

# How many consecutive verdicts may have EVERY member fail before the panel
# gives up. A failed call counts as NO, so during a sustained outage every
# candidate is "rejected" and a builder would burn its batch budget generating
# nothing, then exit successfully with a silently short dataset. Transient
# blips are absorbed by the per-call retry in app/shared/retry.py, so reaching
# this many total failures in a row means the API is genuinely down.
MAX_CONSECUTIVE_TOTAL_FAILURES = 3


class PanelUnavailableError(RuntimeError):
    """Raised when the panel has failed completely too many times in a row.

    Signals "stop and try again later", not "this item is bad". Builders let
    this propagate so the run halts with whatever it has already written to
    disk, which ``--resume`` can continue from.
    """


@dataclass(frozen=True)
class PanelVote:
    """A single panel member's vote.

    ``provider`` records the upstream OpenRouter routed to, so "the panel was
    consistent" is checkable — a slug can be served by several providers at
    different quantizations.
    """

    model: str
    vote: str
    parsed: bool
    provider: str = ""


@dataclass(frozen=True)
class LabelVote:
    """A single panel member's label vote."""

    model: str
    vote: str
    label: str


@dataclass(frozen=True)
class LabelVerdict:
    """Aggregated label verdict from the evaluator panel."""

    votes: List[LabelVote]
    label_counts: Dict[str, int]
    accepted_label: Optional[str]
    accepted: bool
    acceptance_threshold: int


@dataclass(frozen=True)
class PanelVerdict:
    """Aggregated verdict from the evaluator panel.

    ``error_count`` is how many members failed to return a usable vote (after
    retries) and were counted as NO. Non-zero means part of this verdict
    reflects infrastructure rather than the item.
    """

    votes: List[PanelVote]
    yes_count: int
    no_count: int
    accepted: bool
    acceptance_threshold: int
    error_count: int = 0


class EvaluatorPanel:
    """5-model evaluator panel with majority voting.

    Each model independently evaluates a prompt at temperature 0.0.
    An item is accepted if ≥ ``acceptance_threshold`` models vote YES.
    """

    def __init__(self, settings: DatasetGenSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self._models = settings.panel_model_list
        self._threshold = settings.acceptance_threshold
        # Consecutive verdicts in which every member failed — see
        # MAX_CONSECUTIVE_TOTAL_FAILURES. Reset by any verdict that gets at
        # least one real vote.
        self._consecutive_total_failures = 0
        # Models whose reasoning cannot be disabled — for these we omit
        # reasoning:{enabled:false} (sending it would error). Add a slug here
        # only if OpenRouter rejects the flag for it.
        self._reasoning_mandatory: set[str] = set()

    def _track_total_failure(self, error_count: int, panel_size: int) -> None:
        """Trip the circuit breaker after repeated complete panel failures.

        Raises:
            PanelUnavailableError: After MAX_CONSECUTIVE_TOTAL_FAILURES
                consecutive verdicts in which every member failed — stops the
                run rather than letting an outage masquerade as rejected
                candidates.
        """
        if panel_size and error_count == panel_size:
            self._consecutive_total_failures += 1
            if self._consecutive_total_failures >= MAX_CONSECUTIVE_TOTAL_FAILURES:
                logger.error(
                    "panel.unavailable",
                    consecutive_failures=self._consecutive_total_failures,
                    detail="every panel member failed repeatedly; stopping so the run can be resumed later",
                )
                raise PanelUnavailableError(
                    f"All {panel_size} panel members failed on "
                    f"{self._consecutive_total_failures} consecutive items. "
                    "The API is likely unavailable. Rows accepted so far are already "
                    "written to the output CSV — rerun with --resume to continue."
                )
        else:
            self._consecutive_total_failures = 0

    def _build_payload(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        session_id: Optional[str] = None,
    ) -> Dict[str, object]:
        """Build the OpenRouter request payload for a panel model.

        Conditionally includes ``reasoning: {enabled: false}`` for models
        that support disabling reasoning. ``session_id`` is accepted for
        call-site compatibility but NOT sent — caching is automatic and
        prefix-driven, so the field makes no difference.
        """
        payload: Dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": self._settings.panel_temperature,
            "max_tokens": max_tokens,
        }
        if model not in self._reasoning_mandatory:
            payload["reasoning"] = {"enabled": False}

        # ``session_id`` is deliberately NOT added to the payload. Caching is
        # automatic and prefix-driven; the field makes no difference, and
        # OpenRouter silently drops parameters it does not recognise. What
        # does help is keeping the varying part of the prompt LAST so the
        # shared prefix stays byte-identical (already done by the callers),
        # and pinning the provider below.

        # Pin the upstream provider. A single slug can be served by several
        # providers at different quantizations, so temperature=0.0 on its own
        # does not make a panel vote reproducible — a mid-run reroute silently
        # swaps the rater. Keeping calls on one provider is also what actually
        # produces prompt-cache hits on the shared prefix.
        provider: Dict[str, object] = {"allow_fallbacks": self._settings.panel_allow_fallbacks}
        order = [p.strip() for p in self._settings.panel_provider_order.split(",") if p.strip()]
        if order:
            provider["order"] = order
        payload["provider"] = provider
        return payload

    async def evaluate(
        self,
        prompt: str,
        context: str = "",
        session_id: Optional[str] = None,
    ) -> PanelVerdict:
        """Evaluate a draft item using the full panel."""
        tasks = [
            self._evaluate_single(model, prompt, context, session_id)
            for model in self._models
        ]
        votes = await asyncio.gather(*tasks, return_exceptions=True)

        panel_votes: List[PanelVote] = []
        error_count = 0
        for model, result in zip(self._models, votes):
            if isinstance(result, Exception):
                logger.warning(
                    "panel.vote.error",
                    model=model,
                    error=str(result),
                )
                # On error, default to NO (fail-closed)
                error_count += 1
                panel_votes.append(PanelVote(
                    model=model,
                    vote=f"ERROR: {result}",
                    parsed=False,
                ))
            else:
                panel_votes.append(result)

        yes_count = sum(1 for v in panel_votes if v.parsed)
        no_count = len(panel_votes) - yes_count
        accepted = yes_count >= self._threshold

        # A failed call becomes a NO, so enough failures force a rejection no
        # matter what the item says — with 5 members and a threshold of 4, two
        # errors cap the achievable score at 3/5. That is infrastructure noise
        # entering the dataset as a content decision. Retries absorb transient
        # 429/5xx, so reaching this warning means the errors outlived them.
        if error_count and error_count > len(panel_votes) - self._threshold:
            logger.warning(
                "panel.verdict.indeterminate",
                error_count=error_count,
                panel_size=len(panel_votes),
                threshold=self._threshold,
                detail="errors alone could force rejection; treat this verdict as unreliable",
            )

        self._track_total_failure(error_count, len(panel_votes))

        logger.info(
            "panel.verdict",
            yes_count=yes_count,
            no_count=no_count,
            error_count=error_count,
            accepted=accepted,
            threshold=self._threshold,
        )

        return PanelVerdict(
            votes=panel_votes,
            yes_count=yes_count,
            no_count=no_count,
            accepted=accepted,
            acceptance_threshold=self._threshold,
            error_count=error_count,
        )

    @external_api_retry
    async def _evaluate_single(
        self,
        model: str,
        prompt: str,
        context: str,
        session_id: Optional[str] = None,
    ) -> PanelVote:
        """Evaluate with a single panel model."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evaluator. Answer with ONLY 'YES' or 'NO'. "
                    "Be strict and precise."
                ),
            },
            {
                "role": "user",
                "content": f"{prompt}\n\n{context}",
            },
        ]

        response = await self._client.post(
            "/chat/completions",
            json=self._build_payload(model, messages, max_tokens=1024, session_id=session_id),
        )
        response.raise_for_status()
        data = response.json()
        vote_text = self._extract_content(data)

        # Parse YES/NO
        parsed = self._parse_yes_no(vote_text)

        return PanelVote(
            model=model,
            vote=vote_text,
            parsed=parsed,
            provider=str(data.get("provider") or ""),
        )

    @staticmethod
    def _parse_yes_no(text: str) -> bool:
        """Parse a YES/NO response. True if YES, False if NO or ambiguous."""
        text_upper = text.upper().strip()
        # Check for explicit YES
        if re.search(r'\bYES\b', text_upper):
            return True
        # Check for explicit NO
        if re.search(r'\bNO\b', text_upper):
            return False
        # Ambiguous — fail-closed
        return False

    async def evaluate_label(
        self,
        prompt: str,
        context: str,
        valid_labels: List[str],
        session_id: Optional[str] = None,
    ) -> LabelVerdict:
        """Evaluate a draft item by assigning a label (majority voting).

        Each panel model independently assigns one of ``valid_labels``. The
        item is accepted if >= ``acceptance_threshold`` models agree on the
        *same* label. Used by Subset D for sentence-level labeling.

        Args:
            session_id: Pass the same id for every sentence of one question so
                the shared prefix can be cached across per-sentence calls.
        """
        tasks = [
            self._evaluate_label_single(model, prompt, context, valid_labels, session_id)
            for model in self._models
        ]
        votes = await asyncio.gather(*tasks, return_exceptions=True)

        label_votes: List[LabelVote] = []
        for model, result in zip(self._models, votes):
            if isinstance(result, Exception):
                logger.warning(
                    "panel.label_vote.error",
                    model=model,
                    error=str(result),
                )
                label_votes.append(LabelVote(
                    model=model,
                    vote=f"ERROR: {result}",
                    label="",
                ))
            else:
                label_votes.append(result)

        label_counts: Dict[str, int] = {}
        for v in label_votes:
            if v.label:
                label_counts[v.label] = label_counts.get(v.label, 0) + 1

        # A member that errored produces an empty label, so a total outage
        # yields no labels at all and every sentence looks unlabelable. Same
        # circuit breaker as the binary path — stop rather than silently
        # producing a short dataset.
        self._track_total_failure(
            sum(1 for v in label_votes if not v.label), len(label_votes)
        )

        accepted_label: Optional[str] = None
        accepted = False
        if label_counts:
            top_label, top_count = max(label_counts.items(), key=lambda x: x[1])
            if top_count >= self._threshold:
                accepted_label = top_label
                accepted = True

        logger.info(
            "panel.label_verdict",
            label_counts=label_counts,
            accepted_label=accepted_label,
            accepted=accepted,
            threshold=self._threshold,
        )

        return LabelVerdict(
            votes=label_votes,
            label_counts=label_counts,
            accepted_label=accepted_label,
            accepted=accepted,
            acceptance_threshold=self._threshold,
        )

    @external_api_retry
    async def _evaluate_label_single(
        self,
        model: str,
        prompt: str,
        context: str,
        valid_labels: List[str],
        session_id: Optional[str] = None,
    ) -> LabelVote:
        """Assign a label to an item with a single panel model."""
        labels_str = ", ".join(valid_labels)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evaluator. Assign exactly one label from the "
                    f"following list: {labels_str}. "
                    "Respond with ONLY the label name, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"{prompt}\n\n{context}",
            },
        ]

        response = await self._client.post(
            "/chat/completions",
            json=self._build_payload(model, messages, max_tokens=1024, session_id=session_id),
        )
        response.raise_for_status()
        data = response.json()
        vote_text = self._extract_content(data)

        label = self._parse_label(vote_text, valid_labels)

        return LabelVote(model=model, vote=vote_text, label=label)

    @staticmethod
    def _parse_label(text: str, valid_labels: List[str]) -> str:
        """Parse a label response. Returns the matched label or empty string."""
        text_lower = text.lower().strip()
        # Normalize: remove quotes, punctuation, whitespace
        text_clean = re.sub(r'["\'.!,;:]', '', text_lower).strip()
        for label in valid_labels:
            label_lower = label.lower().strip()
            # Exact match or the label appears as a whole word
            if text_clean == label_lower or re.search(r'\b' + re.escape(label_lower) + r'\b', text_clean):
                return label_lower
        return ""

    @staticmethod
    def _extract_content(data: Dict[str, object]) -> str:
        """Extract text content from an OpenRouter chat completion response.

        Handles reasoning models that return ``content: null`` with the text
        in a ``reasoning`` field.
        """
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content")
        if content and isinstance(content, str) and content.strip():
            return content.strip()
        reasoning = message.get("reasoning")
        if reasoning and isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
        return ""

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
