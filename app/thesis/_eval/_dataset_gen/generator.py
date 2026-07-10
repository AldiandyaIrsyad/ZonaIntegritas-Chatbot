"""Dataset generator using DeepSeek V4.

Generates draft items (questions, adversarial inputs, boundary queries, RAM
sentences) from seed prompts. The generator operates at temperature 0.0 for
reproducibility. Drafts are then validated by the EvaluatorPanel.

Usage:
    generator = DatasetGenerator(settings)
    drafts = await generator.generate(
        seed_prompt="Generate 10 factual questions about Zona Integritas...",
        count=10,
    )
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import structlog

from app.thesis._eval._dataset_gen.config import DatasetGenSettings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class GeneratedItem:
    """A single generated draft item.

    Attributes:
        raw: Raw model output.
        parsed: Parsed structured data (dict or string).
    """

    raw: str
    parsed: Any


class DatasetGenerator:
    """LLM-based dataset generator using DeepSeek V4.

    Args:
        settings: DatasetGenSettings with API key and model config.
    """

    def __init__(self, settings: DatasetGenSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        # Models that require reasoning and cannot have it disabled.
        self._reasoning_mandatory = {
            "google/gemini-3.1-pro-preview",
        }

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> Dict[str, object]:
        """Build the OpenRouter request payload for the generator model.

        Conditionally includes ``reasoning: {enabled: false}`` for models
        that support disabling reasoning.

        Args:
            messages: Chat messages.
            max_tokens: Maximum tokens for the response.

        Returns:
            Request payload dict.
        """
        payload: Dict[str, object] = {
            "model": self._settings.generator_model,
            "messages": messages,
            "temperature": self._settings.generator_temperature,
            "max_tokens": max_tokens,
        }
        if self._settings.generator_model not in self._reasoning_mandatory:
            payload["reasoning"] = {"enabled": False}
        return payload

    async def generate(
        self,
        seed_prompt: str,
        count: int = 10,
        system_prompt: Optional[str] = None,
    ) -> List[GeneratedItem]:
        """Generate draft items from a seed prompt.

        Args:
            seed_prompt: Instructions for what to generate.
            count: Number of items to generate.
            system_prompt: Optional system prompt for the generator.

        Returns:
            List of GeneratedItem.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a dataset generator for a RAG evaluation benchmark. "
                "Generate high-quality, diverse items in Indonesian. "
                "Output each item as a JSON object on its own line (JSONL format). "
                "Do not include markdown code fences."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{seed_prompt}\n\nGenerate exactly {count} items."},
        ]

        response = await self._client.post(
            "/chat/completions",
            json=self._build_payload(messages, max_tokens=8192),
        )
        response.raise_for_status()
        data = response.json()
        raw_output = self._extract_content(data)

        items = self._parse_jsonl(raw_output)
        logger.info(
            "dataset_gen.generated",
            requested=count,
            parsed=len(items),
        )
        return items

    @staticmethod
    def _parse_jsonl(text: str) -> List[GeneratedItem]:
        """Parse JSONL output from the generator.

        Args:
            text: Raw model output.

        Returns:
            List of GeneratedItem with parsed JSON.
        """
        items: List[GeneratedItem] = []
        # Remove markdown code fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                items.append(GeneratedItem(raw=line, parsed=parsed))
            except json.JSONDecodeError:
                # Try to extract JSON from the line
                json_match = re.search(r'\{[^}]+\}', line)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        items.append(GeneratedItem(raw=line, parsed=parsed))
                    except json.JSONDecodeError:
                        items.append(GeneratedItem(raw=line, parsed=line))
                else:
                    items.append(GeneratedItem(raw=line, parsed=line))
        return items

    async def generate_single(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a single text response (not JSONL).

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            Raw model response text.
        """
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response = await self._client.post(
            "/chat/completions",
            json=self._build_payload(messages, max_tokens=8192),
        )
        response.raise_for_status()
        data = response.json()
        return self._extract_content(data)

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        """Extract text content from an OpenRouter chat completion response.

        Handles reasoning models that may return ``content: null`` with the
        actual text in a ``reasoning`` field. If both are present, prefers
        ``content``. If both are empty, returns empty string.

        Args:
            data: Parsed JSON response from OpenRouter.

        Returns:
            The response text.
        """
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content")
        if content and content.strip():
            return content
        # Fall back to reasoning field (reasoning models with max_tokens too low)
        reasoning = message.get("reasoning")
        if reasoning and reasoning.strip():
            return reasoning
        return "" if content is None else (content or "")

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
