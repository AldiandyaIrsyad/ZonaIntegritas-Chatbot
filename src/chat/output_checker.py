"""
Output validation and persistence for LLM chat responses.

This module acts as the final stage in the LLM pipeline.  It receives the
fully assembled response and the original user prompt, runs a validation
check (currently a pass-through stub), and persists the assistant message
to the database.

Extension point: replace or augment ``check_and_persist`` with LLM-as-judge,
rule-based filters, or any other output-verification strategy without
touching ``ChatService``.
"""
from typing import Optional

import anyio

from src.core import get_logger
from src.chat.repository import ChatRepository

logger = get_logger(__name__)


async def check_and_persist(
    session_id: str,
    initial_prompt: str,
    final_output: str,
    repository: ChatRepository,
) -> Optional[str]:
    """Validate the LLM output and persist it as an assistant message.

    Currently passes through the output unchanged.  Future implementations
    should replace or extend this function with LLM-as-judge scoring,
    toxicity filtering, factuality checks, or any other output-verification
    strategy.

    The database write is shielded from cancellation via ``anyio.CancelScope``
    to ensure the message is saved even if the client disconnects mid-stream.

    Args:
        session_id: UUID of the chat session the response belongs to.
        initial_prompt: The original user message text.  Available for
            verification strategies that need to compare input and output.
        final_output: The fully assembled LLM response string after all
            streaming chunks have been collected.
        repository: Chat repository used to persist the assistant message.

    Returns:
        Optional[str]: The (possibly modified) output string after checking,
            or ``None`` if ``final_output`` is blank / whitespace-only.
    """
    if not final_output.strip():
        return None

    # ── TODO: LLM-as-judge / output verification hook ───────────────
    # Insert verification logic here.  Example shape:
    #
    #   verdict = await llm_judge.evaluate(initial_prompt, final_output)
    #   if verdict.is_harmful:
    #       return "[Response withheld by safety filter]"
    #
    # For now, the output is accepted as-is.
    verified_output = final_output

    try:
        with anyio.CancelScope(shield=True):
            await repository.create_message(session_id, "assistant", verified_output)
    except Exception:
        logger.error("Failed to persist assistant message", exc_info=True)

    return verified_output
