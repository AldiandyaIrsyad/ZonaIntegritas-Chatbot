"""
Ultimate Orchestrator for the Chat Module.

Coordinates the Knowledge Base, IVM (Safety/Relevance), RAM (Response Assessment),
and the LLM inference engine.
"""

import json
import uuid
import structlog
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

from app.chat.domain.interfaces import IChatRepository, ILLMConnection
from app.kb.application.search_service import SearchService
from app.thesis.ivm.service import IVMService, MaliciousPromptException
from app.thesis.ivm.relevance_service import RelevanceService, IrrelevantQueryException
from app.thesis.prompts import build_prompt
from app.thesis.ram.service import RAMService
from app.thesis.ram.interfaces import RetrievedContext as RAMRetrievedContext
from app.thesis.ram.text_utils import split_sentences_with_seps

logger = structlog.get_logger(__name__)

class ChatService:
    """Orchestrates the chat request pipeline."""

    def __init__(
        self,
        chat_repo: IChatRepository,
        llm_conn: ILLMConnection,
        search_service: SearchService,
        ivm_service: IVMService,
        relevance_service: RelevanceService,
        ram_service: RAMService,
        model_name: str,
        system_prompt: str,
        temperature: float = 0.0,
    ):
        self.chat_repo = chat_repo
        self.llm_conn = llm_conn
        self.search_service = search_service
        self.ivm_service = ivm_service
        self.relevance_service = relevance_service
        self.ram_service = ram_service
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature

    async def create_session(self) -> Dict[str, Any]:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        session = await self.chat_repo.create_session(session_id, "New Chat")
        return {"id": session.id, "title": session.title}

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all chat sessions."""
        sessions = await self.chat_repo.get_all_sessions()
        return [{"id": s.id, "title": s.title} for s in sessions]

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details."""
        session = await self.chat_repo.get_session_by_id(session_id, load_messages=True)
        if not session:
            return None
        return {
            "id": session.id,
            "title": session.title,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "context": m.context,
                    "sources": m.sources,
                }
                for m in session.messages
            ],
        }

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        return await self.chat_repo.delete_session(session_id)

    @staticmethod
    def _format_citation(result: Any) -> str:
        """Format an NLI assessment result into a canonical citation marker.

        Produces a marker of the form
        ``*(STATUS: SCORE; SOURCE; Page N; DocID:ID; Evidence:"snippet")*``
        where STATUS is one of Supported/Contradiction, SCORE is the
        dominant NLI confidence (2 decimals), SOURCE is the document title
        (omitted if empty), Page N is included only when a page number is
        available, DocID:ID is included only when a source document id is
        available (lets the frontend offer a "Download PDF" link), and
        Evidence:"snippet" is included only when a matched sub-passage is
        available (the reranker-matched context window the NLI check was
        actually run against). This format is parsed by the frontend
        ``renderMessage`` function to render colored citation badges with
        tooltips.

        Neutral results (and any unrecognized label, e.g. from the
        disabled/error fallback paths in the NLI adapter) produce no
        marker at all, since there's nothing useful to tell the user.

        Args:
            result: An ``NLIResult`` with ``label``, per-class scores,
                ``source_title``, optional ``page``, and optional ``doc_id``.

        Returns:
            A citation marker string, or an empty string if the result is
            None or the label isn't Supported/Contradiction.
        """
        if result is None:
            return ""

        label_map = {
            "entailment": "Supported",
            "contradiction": "Contradiction",
        }
        status = label_map.get(result.label)
        if status is None:
            return ""

        # Pick the dominant score for the predicted label
        score_map = {
            "entailment": result.entailment_score,
            "contradiction": result.contradiction_score,
        }
        score = score_map[result.label]

        parts = [f"{status}: {score:.2f}"]
        if result.source_title:
            parts.append(result.source_title)
        if result.page is not None:
            parts.append(f"Page {result.page}")
        if result.doc_id:
            parts.append(f"DocID:{result.doc_id}")
        if result.evidence_snippet:
            parts.append(f'Evidence:"{result.evidence_snippet}"')

        return f" *({'; '.join(parts)})*"

    @staticmethod
    def _split_propositions(text: str) -> List[Tuple[str, str]]:
        """Heuristically split a buffer into (proposition, trailing_separator) pairs.

        Splits on standard sentence boundaries and major Indonesian conjunctions
        to evaluate smaller facts independently. The trailing separator is the
        whitespace that followed the proposition in the source text (e.g.
        "\\n\\n" for a paragraph break), so callers can preserve the original
        formatting instead of always rejoining with a single space.
        """
        import re

        # Split on sentence boundaries (., ?, !) and newlines. Delegates to
        # the shared splitter, which guards against markdown list markers
        # (e.g. "1.", "2.") being mistaken for sentence ends, and which
        # reports the exact separator matched at each boundary.
        sentence_pairs = split_sentences_with_seps(text)

        propositions: List[Tuple[str, str]] = []
        for part, sep in sentence_pairs:
            # Split on Indonesian conjunction markers that introduce new claims,
            # keeping the conjunction with the second part if possible (handled by split, we'll re-attach or just let NLI handle it).
            # Using positive lookbehind to keep the delimiter attached to the right part.
            sub_parts = re.split(r'(?i)(,\s*yang\s+|,\s*dan\s+|,\s*karena\s+|,\s*sehingga\s+)', part)

            subs = []
            current_prop = ""
            for sp in sub_parts:
                if re.match(r'(?i)(,\s*yang\s+|,\s*dan\s+|,\s*karena\s+|,\s*sehingga\s+)', sp):
                    # It's a delimiter, start a new proposition with it
                    if current_prop.strip():
                        subs.append(current_prop.strip())
                    current_prop = sp.lstrip(", ") # Strip leading comma for cleaner premise
                else:
                    current_prop += sp

            if current_prop.strip():
                subs.append(current_prop.strip())

            # Conjunction splits are always mid-line, so only the last
            # sub-proposition of a sentence carries the sentence's real
            # trailing separator (e.g. a paragraph break); earlier ones
            # just get a plain space.
            for i, sub in enumerate(subs):
                propositions.append((sub, sep if i == len(subs) - 1 else " "))

        return propositions

    async def process_chat_message(
        self, session_id: str, message_text: str, skip_guardrails: bool = False
    ) -> AsyncGenerator[str, None]:
        """The main generation pipeline: Safety -> Pre-check -> Context -> Generate -> Assess.

        When ``skip_guardrails`` is True, the IVM safety/relevance checks and
        the RAM per-sentence assessment are bypassed (baseline mode for
        Experiment 4). Retrieval still runs so the LLM has context.
        """
        
        # 1. Initialize or get session
        session = await self.chat_repo.get_session_by_id(session_id, load_messages=True)
        is_new_session = False
        if not session:
            session = await self.chat_repo.create_session(session_id, message_text[:20] + "...")
            is_new_session = True
        elif session.title == "New Chat":
            new_title = message_text[:30] + ("..." if len(message_text) > 30 else "")
            await self.chat_repo.update_session_title(session, new_title)

        # Capture history BEFORE any DB writes to avoid lazy-load issues
        # in async context (greenlet_spawn errors after flush/commit).
        # For newly created sessions, messages is empty (no lazy load needed).
        if is_new_session:
            history: List = []
        else:
            history = list(session.messages[-10:]) if session and session.messages else []

        # Record user message
        await self.chat_repo.create_message(session_id, "user", message_text, raw_content=message_text)

        try:
            # 2. Safety Check (IVM)
            if not skip_guardrails:
                await self.ivm_service.check_malicious(message_text)

            # 3. Pre-check Relevance (IVM + KB)
            # NOTE: We intentionally do NOT pass session_id here. The chat
            # session ID is unrelated to the KB chunk session_id payload —
            # KB chunks are ingested without a session_id and the Qdrant
            # filter would return zero results if we passed the chat
            # session ID.
            precheck_contexts = await self.search_service.search(message_text, top_k=3)
            if not skip_guardrails:
                if not precheck_contexts:
                    raise IrrelevantQueryException("No relevant contexts found in the knowledge base.")

                context_chunks = [ctx.text for ctx in precheck_contexts]
                context_scores = [ctx.score for ctx in precheck_contexts]
                await self.relevance_service.check_relevance(message_text, context_chunks, context_scores)

            # 4. Deep Context Retrieval (KB)
            full_contexts = await self.search_service.search(message_text, top_k=15)
            
            # Map KB contexts to RAM contexts (preserve breadcrumbs + hierarchy)
            ram_contexts = [
                RAMRetrievedContext(
                    text=ctx.text,
                    source_title=ctx.source_title,
                    page=ctx.page,
                    breadcrumbs=ctx.breadcrumbs,
                    content_type=ctx.content_type,
                    chunk_id=ctx.chunk_id,
                    path=ctx.path,
                    doc_id=ctx.doc_id,
                )
                for ctx in full_contexts
            ]

            # 5. Prompt Building (Indonesian system/context prompt + a
            # cryptographically random per-request delimiter wrapping the
            # raw user message, as an additional defense against system
            # prompt injection on top of the IVM safety check above).
            bundle = build_prompt(message_text, ram_contexts, self.system_prompt)

            messages = [{"role": "system", "content": bundle.system_prompt}]
            # Add history (up to last 5 messages to avoid blowing up context window)
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

            # Add current
            messages.append({"role": "user", "content": bundle.user_turn})

            # Prepare RAM evaluation
            premise = self.ram_service.build_premise(ram_contexts)

            # Emit retrieved context so downstream consumers (e.g. Subset D
            # dataset generation) can capture the source passages used for
            # generation, and so the frontend can render it in a collapsible
            # "view RAG context" panel. This is emitted as a single NDJSON
            # event before streaming begins, and the same payload is
            # persisted alongside the assistant message below so it
            # survives a page refresh.
            #
            # "content" is a flat join of all chunk texts and MUST keep this
            # exact shape: app/thesis/_eval/_dataset_gen/build_subset_d.py
            # reads it verbatim as the RAM ground-truth retrieved_context.
            # "chunks" is the richer, per-source structure the chat UI uses
            # to pair each chunk's page/section with its own text (rather
            # than a page list followed by one disconnected text blob).
            context_payload = {
                "content": "\n\n".join(ctx.text for ctx in ram_contexts),
                "chunks": [
                    {
                        "title": ctx.source_title,
                        "page": ctx.page,
                        "breadcrumbs": ctx.breadcrumbs,
                        "text": ctx.text,
                    }
                    for ctx in ram_contexts
                ],
            }
            yield json.dumps({"type": "context", **context_payload}) + "\n"

            # We buffer the stream by sentence to evaluate each complete sentence
            buffer = ""
            final_output = ""

            # 6. Stream and Assess
            stream = self.llm_conn.stream_chat(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=self.temperature,
            )
            
            async for chunk in stream:
                buffer += chunk
                # Check if we hit a boundary that likely ends a proposition
                if any(punct in chunk for punct in [". ", "? ", "! ", "\n", ", yang ", ", dan ", ", karena ", ", sehingga "]):
                    propositions = self._split_propositions(buffer)
                    if len(propositions) > 1:
                        # Process all but the last incomplete fragment
                        for prop, sep in propositions[:-1]:
                            prop_text = prop.strip()
                            # Re-add trailing period if it was stripped by regex and wasn't a conjunction split,
                            # or just ensure it's a complete looking sentence.
                            if not prop_text.endswith((".", "?", "!")):
                                prop_text += "."

                            if len(prop_text) > 15:  # Only assess meaningful propositions
                                if skip_guardrails:
                                    # Baseline mode: no RAM assessment, no citation
                                    out_prop = prop_text
                                else:
                                    result = await self.ram_service.assess_sentence(prop_text, premise, ram_contexts)
                                    out_prop = prop_text + self._format_citation(result)
                                final_output += out_prop + sep
                                yield json.dumps({"type": "chunk", "content": out_prop + sep}) + "\n"
                            else:
                                final_output += prop_text + sep
                                yield json.dumps({"type": "chunk", "content": prop_text + sep}) + "\n"

                        buffer = propositions[-1][0]

            # Process any remaining buffer
            if buffer.strip():
                sentence = buffer.strip()
                if len(sentence) > 10 and not skip_guardrails:
                    result = await self.ram_service.assess_sentence(sentence, premise, ram_contexts)
                    sentence = sentence + self._format_citation(result)
                final_output += sentence
                yield json.dumps({"type": "chunk", "content": sentence}) + "\n"

            # Record assistant message, alongside the RAG context/sources
            # used to generate it so the frontend can restore the "view RAG
            # context" panel after a page refresh.
            await self.chat_repo.create_message(
                session_id, "assistant", final_output, raw_content=final_output,
                context=context_payload["content"], sources=context_payload["chunks"],
            )
            yield json.dumps({"type": "done"}) + "\n"

        except MaliciousPromptException:
            err_msg = "Your request was blocked by our safety filters."
            await self.chat_repo.create_message(session_id, "assistant", err_msg)
            yield json.dumps({"type": "error", "content": err_msg}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except IrrelevantQueryException:
            err_msg = "I'm sorry, but I can only answer questions related to the provided knowledge base documents."
            await self.chat_repo.create_message(session_id, "assistant", err_msg)
            yield json.dumps({"type": "error", "content": err_msg}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            logger.error("chat.pipeline.failed", error=str(e), exc_info=True)
            yield json.dumps({"type": "error", "content": "An unexpected error occurred."}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
