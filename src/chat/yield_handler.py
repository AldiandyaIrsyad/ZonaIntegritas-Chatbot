"""
NLI-annotated streaming handler for LLM responses.

Implements a sentence-buffered double-streaming pipeline:

  LLM tokens → sentence buffer
            → [boundary hit] asyncio.Task(NLI) queued
            → emit sentence only after its NLI task is done (FIFO)
            → annotate contradicted / supported sentences inline

The LLM stream is never paused waiting for NLI — tasks run concurrently
and sentences are released as soon as their individual task completes,
preserving strict emission order.

Usage::

    async for chunk in nli_streaming_generate(
        raw_history, contexts, ram_service, llm_service
    ):
        yield chunk
        response_content += chunk
"""
import asyncio
import re
from typing import AsyncIterator, List

from src.infra.nli import NLIResult

from src.llm import LLMService
from src.ram.service import RAMService
from src.rag import RetrievedContext


async def nli_streaming_generate(
    raw_history: List[dict[str, str]],
    contexts: List[RetrievedContext],
    ram_service: RAMService,
    llm_service: LLMService,
) -> AsyncIterator[str]:
    """Yield NLI-annotated chunks from the LLM stream in FIFO sentence order.

    Each sentence is dispatched asynchronously to the NLI service as soon as
    a sentence boundary is detected; the sentence is only emitted once its
    NLI task completes.  This keeps the LLM stream flowing without blocking.

    Annotation format appended to each sentence:
        - Entailment:    ``*(Supported: 0.93)*``
        - Contradiction: ``*(Contradiction: 0.87)*``
        - Neutral:       ``*(Neutral: 0.61)*``

    Args:
        raw_history: Full conversation history including the system prompt
            as the first element, ready to be forwarded to the LLM.
        contexts: RAG knowledge-base contexts used to build the NLI premise.
            Only KB contexts are used; session PDFs are treated as user intent.
        ram_service: The RAM/NLI service instance used for sentence assessment.
        llm_service: The LLM streaming service.

    Yields:
        str: Annotated text chunks in strict FIFO order, suitable for direct
            streaming to the client.
    """
    # Sentence-boundary regex: period / ! / ? / newlines followed by whitespace.
    # The trailing whitespace is included so spaces between sentences are preserved.
    _BOUNDARY = re.compile(r"[.!?\n\u3002]+\s+")

    # Build the NLI premise once for the entire request (not per sentence).
    premise = ram_service.build_premise(contexts)

    def _start_nli(sentence: str) -> "asyncio.Task[NLIResult]":
        """Kick off NLI assessment for *sentence* without blocking.

        Args:
            sentence: The sentence text to assess.

        Returns:
            asyncio.Task[NLIResult]: A running task whose result is an NLIResult.
        """
        return asyncio.create_task(ram_service.assess_sentence(sentence, premise))

    def _annotate(raw_sentence: str, task: "asyncio.Task[NLIResult]") -> str:
        """Apply NLI annotation to *raw_sentence* using the completed *task*.

        Args:
            raw_sentence: Original sentence text.
            task: Completed asyncio.Task[NLIResult] containing an NLIResult.

        Returns:
            str: Sentence with inline NLI annotation appended.
        """
        result: NLIResult = task.result()
        if result.label == "entailment":
            return raw_sentence + f" *(Supported: {result.entailment_score:.2f})*"
        elif result.label == "contradiction":
            return raw_sentence + f" *(Contradiction: {result.contradiction_score:.2f})*"
        else:
            return raw_sentence + f" *(Neutral: {result.neutral_score:.2f})*"

    sentence_buffer = ""
    # pending: list of (raw_sentence_text, asyncio.Task[NLIResult])
    pending: list[tuple[str, asyncio.Task[NLIResult]]] = []

    try:
        async for token in llm_service.stream_response(raw_history):
            sentence_buffer += token

            # Drain ALL complete sentences already in the buffer.
            while match := _BOUNDARY.search(sentence_buffer):
                end_idx = match.end()
                sentence = sentence_buffer[:end_idx]
                sentence_buffer = sentence_buffer[end_idx:]
                if sentence.strip():
                    pending.append((sentence, _start_nli(sentence)))

            # Drain the front of the queue: emit any sentences whose
            # NLI task has already finished (strict FIFO order).
            while pending and pending[0][1].done():
                raw_sent, task = pending.pop(0)
                yield _annotate(raw_sent, task)

        # LLM stream done — split remaining buffer on every boundary,
        # dispatching a separate NLI task per complete sentence, then
        # append the tail (no trailing boundary) as the final item.
        while match := _BOUNDARY.search(sentence_buffer):
            end_idx = match.end()
            sentence = sentence_buffer[:end_idx]
            sentence_buffer = sentence_buffer[end_idx:]
            if sentence.strip():
                pending.append((sentence, _start_nli(sentence)))
        if sentence_buffer.strip():
            pending.append((sentence_buffer, _start_nli(sentence_buffer)))

        # Await remaining tasks in FIFO order and emit.
        for raw_sent, task in pending:
            await task
            yield _annotate(raw_sent, task)
    except (GeneratorExit, Exception):
        for _, task in pending:
            task.cancel()
        raise