"""Evaluator panel for dataset generation.

Implements the 5-model majority voting panel defined in skripsi §3.2.1c.
Each panel member independently evaluates a draft item at temperature 0.0.
An item is accepted if ≥4/5 members agree.

Usage:
    panel = EvaluatorPanel(settings)
    verdict = await panel.evaluate(
        prompt="Is this question answerable from the given context? Answer YES or NO.",
        context="Context: ... Question: ...",
    )
    if verdict.accepted:
        ...
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
import structlog

from app.thesis._eval._dataset_gen.config import DatasetGenSettings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PanelVote:
    """A single panel member's vote.

    Attributes:
        model: Model identifier.
        vote: The raw vote text from the model.
        parsed: Parsed binary verdict (True = YES/accept, False = NO/reject).
    """

    model: str
    vote: str
    parsed: bool


@dataclass(frozen=True)
class LabelVote:
    """A single panel member's label vote.

    Attributes:
        model: Model identifier.
        vote: The raw vote text from the model.
        label: Parsed label string (lowercased), or empty if unparseable.
    """

    model: str
    vote: str
    label: str


@dataclass(frozen=True)
class LabelVerdict:
    """Aggregated label verdict from the evaluator panel.

    Attributes:
        votes: List of individual label votes.
        label_counts: Mapping of label -> count.
        accepted_label: The majority label if >= threshold agreed, else None.
        accepted: True if a label reached the acceptance threshold.
        acceptance_threshold: The threshold used.
    """

    votes: List[LabelVote]
    label_counts: Dict[str, int]
    accepted_label: Optional[str]
    accepted: bool
    acceptance_threshold: int


@dataclass(frozen=True)
class PanelVerdict:
    """Aggregated verdict from the evaluator panel.

    Attributes:
        votes: List of individual panel votes.
        yes_count: Number of YES votes.
        no_count: Number of NO votes.
        accepted: True if yes_count >= acceptance_threshold.
        acceptance_threshold: The threshold used.
    """

    votes: List[PanelVote]
    yes_count: int
    no_count: int
    accepted: bool
    acceptance_threshold: int


class EvaluatorPanel:
    """5-model evaluator panel with majority voting.

    Each model independently evaluates a prompt at temperature 0.0.
    An item is accepted if ≥ ``acceptance_threshold`` models vote YES.

    Args:
        settings: DatasetGenSettings with API key and model list.
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

    async def evaluate(
        self,
        prompt: str,
        context: str = "",
    ) -> PanelVerdict:
        """Evaluate a draft item using the full panel.

        Args:
            prompt: Evaluation prompt (instructions for the panel).
            context: The draft item to evaluate (appended to prompt).

        Returns:
            PanelVerdict with individual votes and acceptance decision.
        """
        tasks = [
            self._evaluate_single(model, prompt, context)
            for model in self._models
        ]
        votes = await asyncio.gather(*tasks, return_exceptions=True)

        panel_votes: List[PanelVote] = []
        for model, result in zip(self._models, votes):
            if isinstance(result, Exception):
                logger.warning(
                    "panel.vote.error",
                    model=model,
                    error=str(result),
                )
                # On error, default to NO (fail-closed)
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

        logger.info(
            "panel.verdict",
            yes_count=yes_count,
            no_count=no_count,
            accepted=accepted,
            threshold=self._threshold,
        )

        return PanelVerdict(
            votes=panel_votes,
            yes_count=yes_count,
            no_count=no_count,
            accepted=accepted,
            acceptance_threshold=self._threshold,
        )

    async def _evaluate_single(
        self,
        model: str,
        prompt: str,
        context: str,
    ) -> PanelVote:
        """Evaluate with a single panel model.

        Args:
            model: Model identifier.
            prompt: Evaluation prompt.
            context: Draft item context.

        Returns:
            PanelVote with the model's verdict.
        """
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
            json={
                "model": model,
                "messages": messages,
                "temperature": self._settings.panel_temperature,
                "max_tokens": 10,
            },
        )
        response.raise_for_status()
        data = response.json()
        vote_text = data["choices"][0]["message"]["content"].strip()

        # Parse YES/NO
        parsed = self._parse_yes_no(vote_text)

        return PanelVote(model=model, vote=vote_text, parsed=parsed)

    @staticmethod
    def _parse_yes_no(text: str) -> bool:
        """Parse a YES/NO response from the model.

        Args:
            text: Raw model response.

        Returns:
            True if YES, False if NO or ambiguous.
        """
        text_upper = text.upper().strip()
        # Check for explicit YES
        if re.search(r'\bYES\b', text_upper):
            return True
        # Check for explicit NO
        if re.search(r'\bNO\b', text_upper):
            return False
        # Ambiguous — fail-closed
        return False

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
