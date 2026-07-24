"""Hosted generative safety-guard infrastructure adapter.

Infra adapter for the IVM (Input Validation Module) research core. Calls
Qwen3Guard-Gen over an OpenAI-compatible chat-completions API and reduces its
generated verdict to the binary decision the IVM needs.

Fulfills: ``app/thesis/ivm/interfaces.py::ISafetyModel``.
Wired in: ``app/chat/dependency.py::get_safety_model``.

Why this exists alongside ``PromptGuardClient``
-----------------------------------------------
The local classifier and this model answer *different questions*, and the
difference is the point of comparing them.

``Llama-Prompt-Guard-2`` flags a prompt only when it explicitly attempts to
override prior instructions, regardless of whether the prompt is harmful, and
Indonesian is not among the eight languages its authors evaluated. Qwen3Guard
was trained on natively-collected Indonesian and classifies against a content
policy that includes a ``Jailbreak`` category — so its scope is much closer to
this system's actual policy, which RQ2 frames as "input berbahaya" rather than
prompt injection narrowly.

Why it is not the default
-------------------------
It cannot be deployed here. On the target machine — an 8 GB laptop GPU already
holding the reranker, the NLI model and the in-process embedder — the smallest
Qwen3Guard is 751M parameters against Prompt Guard 2's 86M, and the 4B and 8B
variants exceed the whole card in half precision. This adapter therefore calls a
hosted endpoint, and is intended for evaluation rather than for the per-turn
chat path.

Output contract
---------------
Qwen3Guard-Gen generates text of the form::

    Safety: Unsafe
    Categories: Jailbreak

Three tiers, not two. ``Controversial`` is mapped by configuration rather than
by a constant here, because that mapping is a methodological choice that has to
be fixed before an experiment runs.
"""

from __future__ import annotations

import re
from typing import List, Optional

import httpx
import structlog

from app.thesis.ivm.interfaces import ISafetyModel, SafetyResult

logger = structlog.get_logger(__name__)

# The generated verdict is a fixed two-line form; parsing it with a regex is
# appropriate here precisely because the model was trained to emit exactly this.
SAFETY_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
CATEGORY_PATTERN = re.compile(r"Categories?:\s*(.+)", re.IGNORECASE)


class Qwen3GuardClient(ISafetyModel):
    """Infrastructure adapter for hosted generative safety classification.

    Fulfills: ``app/thesis/ivm/interfaces.py::ISafetyModel``.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        controversial_is_unsafe: bool = True,
        timeout: float = 30.0,
    ):
        """Configure the OpenAI-compatible client used for classification.

        Args:
            base_url: API base URL, e.g. ``https://router.huggingface.co/v1``.
            api_key: Bearer token for the endpoint.
            model: Model identifier, e.g.
                ``Qwen/Qwen3Guard-Gen-0.6B:featherless-ai``.
            controversial_is_unsafe: Whether the ``Controversial`` tier counts
                as unsafe. True matches the fail-closed posture of the rest of
                the IVM.
            timeout: Request timeout in seconds. Higher than the local
                classifier's because this is a network call to a generative
                model rather than a single forward pass.
        """
        self.model = model
        self.controversial_is_unsafe = controversial_is_unsafe
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        logger.info(
            "chat.qwen3guard.initialized",
            model=model,
            base_url=base_url,
            controversial_is_unsafe=controversial_is_unsafe,
        )

    @staticmethod
    def parse_verdict(content: str) -> tuple[Optional[str], List[str]]:
        """Extract the safety tier and categories from the generated verdict.

        Args:
            content: Raw model output.

        Returns:
            A tuple of (tier or None when unparseable, category names).
        """
        tier_match = SAFETY_PATTERN.search(content or "")
        tier = tier_match.group(1).capitalize() if tier_match else None

        categories: List[str] = []
        category_match = CATEGORY_PATTERN.search(content or "")
        if category_match:
            categories = [
                c.strip()
                for c in category_match.group(1).split(",")
                if c.strip() and c.strip().lower() != "none"
            ]
        return tier, categories

    async def check_prompt(self, text: str) -> SafetyResult:
        """Fulfills ``ISafetyModel.check_prompt``: classify ``text`` by asking a
        hosted Qwen3Guard-Gen model for a safety verdict and reducing its three
        tiers to a binary decision.

        Fails closed: a request error, an empty response, or a verdict that
        cannot be parsed returns ``is_safe=False`` rather than letting unchecked
        input through. An unparseable verdict is treated as unsafe rather than
        safe because this adapter sits on the same gate as the local
        classifier, where the cost of a wrong "safe" is unbounded and the cost
        of a wrong "unsafe" is one rejected turn.
        """
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": text}],
                    # The verdict is two short lines; anything longer means the
                    # model has drifted off its trained output format.
                    "max_tokens": 64,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices:
                logger.warning("chat.qwen3guard.empty_response", model=self.model)
                return SafetyResult(is_safe=False, message="Service unavailable")

            content = (choices[0].get("message") or {}).get("content", "") or ""
            tier, categories = self.parse_verdict(content)

            if tier is None:
                logger.warning(
                    "chat.qwen3guard.unparseable_verdict",
                    model=self.model,
                    raw=content[:200],
                )
                return SafetyResult(is_safe=False, message="Unparseable verdict")

            unsafe = tier == "Unsafe" or (
                tier == "Controversial" and self.controversial_is_unsafe
            )
            logger.debug(
                "chat.qwen3guard.prediction",
                tier=tier,
                categories=categories,
                unsafe=unsafe,
            )

            if unsafe:
                detail = f" ({', '.join(categories)})" if categories else ""
                return SafetyResult(
                    is_safe=False,
                    message=f"Policy violation: {tier}{detail}",
                )
            return SafetyResult(is_safe=True, message=f"Safe ({tier})")

        except Exception as e:
            logger.error("chat.qwen3guard.failed", model=self.model, error=str(e))
            return SafetyResult(is_safe=False, message="Service unavailable")

    async def close(self) -> None:
        """Release the underlying ``httpx.AsyncClient`` connection pool."""
        await self._client.aclose()
