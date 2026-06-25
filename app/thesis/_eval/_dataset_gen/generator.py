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
            json={
                "model": self._settings.generator_model,
                "messages": messages,
                "temperature": self._settings.generator_temperature,
                "max_tokens": 4096,
            },
        )
        response.raise_for_status()
        data = response.json()
        raw_output = data["choices"][0]["message"]["content"]

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
            json={
                "model": self._settings.generator_model,
                "messages": messages,
                "temperature": self._settings.generator_temperature,
                "max_tokens": 2048,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
