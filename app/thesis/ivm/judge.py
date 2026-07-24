"""
LLM-based Judge implementation for relevance checking.

Fulfills: ``app/thesis/ivm/interfaces.py::IJudge``.
Wired in: ``app/chat/dependency.py::get_relevance_checker`` (constructs the
``LLMJudge`` and wraps it in ``app/thesis/ivm/checkers.py::LLMJudgeRelevanceChecker``
when ``ChatConfig.ood_method == "llm_judge"``).

Depends only on the narrow ``ILLMJudgeConnection`` Protocol, not any
concrete HTTP client, so this stays part of the infra-free ``thesis``
research core (see ``docs/02-arsitektur.md`` §2.2).
"""
from typing import Optional

import structlog
from app.thesis.ivm.interfaces import IJudge, ILLMJudgeConnection

logger = structlog.get_logger(__name__)

DEFAULT_RELEVANCE_JUDGE_PROMPT = (
    "You are a relevance judge for a retrieval-augmented QA system. Your "
    "task is to determine if a given query is on the same topic or domain "
    "as the provided context, even if the context does not fully or "
    "directly answer it. "
    "Reply with exactly 'YES' if the query relates to the same subject "
    "matter as the context. "
    "Reply with exactly 'NO' only if the query is about a clearly "
    "unrelated topic, or is malicious/nonsensical."
)

DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE = (
    "Context:\n{context}\n\nQuery: {query}\n\nIs this relevant?"
)


class LLMJudge(IJudge):
    """Judge that uses an LLM to evaluate relevance.

    Fulfills: ``app/thesis/ivm/interfaces.py::IJudge``.
    Wired in: ``app/chat/dependency.py::get_relevance_checker``.
    """

    def __init__(
        self,
        llm_connection: ILLMJudgeConnection,
        model: str = "llama3-70b-8192",
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> None:
        """Args:
            llm_connection: Narrow LLM connection (``ILLMJudgeConnection``,
                ``stream_chat`` only) used to run the judge prompt.
            model: Model identifier passed through to ``stream_chat``.
            system_prompt: Overrides ``DEFAULT_RELEVANCE_JUDGE_PROMPT`` if given.
            user_template: Overrides ``DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE``
                if given. Must contain ``{context}`` and ``{query}`` placeholders.
        """
        self.llm_connection = llm_connection
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_RELEVANCE_JUDGE_PROMPT
        self.user_template = user_template or DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE

    async def evaluate_relevance(self, query: str, context: str) -> bool:
        """Ask the LLM whether ``query`` is on-topic for ``context``.

        Streams the judge's response (see ``max_tokens`` comment below for
        why 400, not a short cap) and fail-closed-parses the trailing
        YES/NO verdict.

        Args:
            query: The user's raw query text.
            context: Joined KB context chunks to judge relevance against.

        Returns:
            bool: True only if the model's last word starts with "YES".

        Raises:
            Exception: Re-raised from the LLM connection on failure — the
                caller (``LLMJudgeRelevanceChecker``) treats this as
                irrelevant (fail closed).
        """
        user_content = self.user_template.replace("{context}", context).replace("{query}", query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        logger.info("llm_judge.evaluating", query=query[:100])
        try:
            response_chunks = []
            async for chunk in self.llm_connection.stream_chat(
                model=self.model,
                messages=messages,
                # Reasoning models (e.g. Qwen3-32B) don't reliably honor
                # LLMConnection's reasoning-disable request over streaming:
                # confirmed live that the actual OpenRouter response cleanly
                # separates message.content ("NO") from a ~240-token
                # message.reasoning field in non-streaming mode, but
                # LLMConnection.stream_chat's fallback (yield the reasoning
                # delta whenever the content delta is empty, so callers that
                # just concatenate chunks don't silently lose everything)
                # means the reasoning preamble arrives as regular chunks
                # *before* the real answer. 50 tokens was nowhere near
                # enough to get past ~240 tokens of reasoning to the actual
                # one-word answer, so the judge always saw only reasoning
                # text and (correctly, per its own fail-closed logic) never
                # matched "YES" — silently blocking every query regardless
                # of true relevance. 400 gives headroom past a typical
                # reasoning preamble.
                max_tokens=400,
            ):
                response_chunks.append(chunk)

            response_text = "".join(response_chunks).strip().upper()
            logger.info("llm_judge.result", query=query[:100], response=response_text, chunk_count=len(response_chunks))

            # Fail closed: if we don't get a clear YES, it's irrelevant.
            # Check the *last* word rather than requiring the response to
            # start with YES/NO — reasoning models conclude with the answer
            # after their preamble rather than leading with it (non-reasoning
            # models just output "YES"/"NO" directly, where first-word and
            # last-word checks are equivalent, so this doesn't regress them).
            last_word = response_text.split()[-1].rstrip(".,!?'\"") if response_text.split() else ""
            return last_word.startswith("YES")
        except Exception as e:
            logger.error("llm_judge.error", error=str(e), exc_info=True)
            # Fail closed on exception
            raise
