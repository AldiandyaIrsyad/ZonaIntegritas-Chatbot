"""
Ultimate Orchestrator for the Chat Module.

Coordinates the Knowledge Base, IVM (Safety/Relevance), RAM (Response Assessment),
and the LLM inference engine.
"""

import json
import re
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

# A Markdown table row (header, separator, or data row) starts and ends
# with "|". Used to keep citation markers off table row lines — appending
# text after a row's closing "|" would corrupt GFM table syntax.
_TABLE_ROW_RE = re.compile(r'^\s*\|.*\|\s*$')

# Minimum proposition length (chars) worth an NLI call. Short factual
# completions (dates, amounts, short names) are common in this domain and
# should still get a citation attempt, so this is intentionally low —
# it mainly filters pure interjections ("Ya.", "Oke.").
MIN_ASSESSABLE_LENGTH = 8

class ChatService:
    """Orchestrates the chat request pipeline (application-layer service).

    ``process_chat_message`` runs the full streaming RAG pipeline for a
    single user turn, in order:

        1. Persist the incoming user message (and create/rename the session).
        2. Safety check the combined message+attachment text (IVM).
        3. Pre-check topical relevance against a shallow KB search (IVM).
        4. Deep KB retrieval for generation context (top_k=15).
        5. Build the system/user prompt (with anti-injection delimiter).
        6. Stream the LLM completion, buffering by sentence/proposition.
        7. Per-proposition, assess entailment against the retrieved context
           and append a citation marker (RAM) — skipped in baseline mode.
        8. Persist the final assistant message alongside its RAG context.

    Collaborators injected at construction (each a port from
    ``app/chat/domain/interfaces.py`` or a service from another bounded
    context, wired in ``app/chat/dependency.py::get_chat_service``):
        - ``chat_repo``: ``IChatRepository`` — session/message persistence.
        - ``llm_conn``: ``ILLMConnection`` — streaming/non-streaming LLM calls.
        - ``search_service``: ``app.kb.application.search_service.SearchService``
          — hybrid KB retrieval (used for both the relevance pre-check and
          the deep context fetch).
        - ``ivm_service``: ``app.thesis.ivm.service.IVMService`` — the Input
          Validation Module; prompt-injection/malicious-content safety gate.
        - ``relevance_service``: ``app.thesis.ivm.relevance_service.RelevanceService``
          — the IVM's out-of-domain / topical relevance gate.
        - ``ram_service``: ``app.thesis.ram.service.RAMService`` — the
          Response Assessment Module; per-sentence NLI entailment check
          used to attach hallucination-aware citation markers.

    Each defense can be disabled independently for ablation experiments —
    ``skip_ivm`` (steps 2-3), ``skip_ram`` (step 7), and ``skip_nonce`` (the
    delimiter in step 5) — with ``skip_guardrails`` kept as the shorthand that
    sets the first two together. Retrieval always runs so the LLM has context.
    See ``process_chat_message``.
    """

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
        attachment_search_excerpt_chars: int = 4000,
    ):
        """Wire the pipeline's collaborators and generation settings.

        Args:
            chat_repo: Session/message persistence port.
            llm_conn: LLM connection port used for streaming generation.
            search_service: KB hybrid retrieval service (relevance
                pre-check + deep context fetch).
            ivm_service: Safety (prompt-injection) checker.
            relevance_service: Topical/OOD relevance checker.
            ram_service: Per-sentence NLI assessment service used to
                generate citation markers.
            model_name: LLM model identifier passed to ``llm_conn``.
            system_prompt: Base system prompt template (see
                ``app.thesis.prompts.build_prompt``).
            temperature: Sampling temperature for generation.
            attachment_search_excerpt_chars: Max characters of an uploaded
                attachment's text folded into the KB search query (the full
                text still goes to the LLM prompt) — see
                ``ChatConfig.attachment_search_excerpt_chars``.
        """
        self.chat_repo = chat_repo
        self.llm_conn = llm_conn
        self.search_service = search_service
        self.ivm_service = ivm_service
        self.relevance_service = relevance_service
        self.ram_service = ram_service
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.attachment_search_excerpt_chars = attachment_search_excerpt_chars

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
                    "attachment_filename": m.attachment_filename,
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

    @staticmethod
    def _is_table_row(prop_text: str) -> bool:
        """True if a proposition is a single Markdown table line (header,
        separator, or data row) — starts and ends with '|'.

        Used to keep citation markers off table row lines, which would
        corrupt GFM table syntax if appended inline (a valid row must
        consist solely of pipe-delimited cells).

        Known limitation: a bare-pipe-notation sentence like "|x| = 5"
        would false-positive; accepted given this domain (Indonesian
        regulatory/thesis prose essentially never uses that notation).
        """
        return bool(_TABLE_ROW_RE.match(prop_text.strip()))

    async def _assess_and_format(
        self,
        text: str,
        premise: str,
        ram_contexts: List[RAMRetrievedContext],
        skip_ram: bool,
    ) -> str:
        """Run RAM assessment (unless in baseline mode) and format the
        result into a citation marker string (possibly "").

        Shared by the per-proposition prose path and the accumulated
        table-block path in ``_handle_complete_proposition``.
        """
        if skip_ram:
            return ""
        result = await self.ram_service.assess_sentence(text, premise, ram_contexts)
        return self._format_citation(result)

    async def _handle_complete_proposition(
        self,
        prop_text: str,
        sep: str,
        *,
        ram_contexts: List[RAMRetrievedContext],
        premise: str,
        skip_ram: bool,
        table_rows: List[Tuple[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Process one complete (proposition, sep) pair, yielding output
        text chunks in the order they should be streamed/appended.

        Table row propositions are accumulated in the caller-owned
        ``table_rows`` list (mutated in place) instead of being assessed
        individually — a single row has no column headers, and injecting
        a citation marker onto a row's line would corrupt GFM table
        syntax. Once a non-table-row proposition arrives (the table block
        has ended), the accumulated rows are assessed as one unit and the
        citation is emitted as its own paragraph, never on a row line.
        """
        if self._is_table_row(prop_text):
            table_rows.append((prop_text, sep))
            yield prop_text + sep
            return

        if table_rows:
            table_text = "\n".join(row for row, _ in table_rows)
            table_rows.clear()
            citation = await self._assess_and_format(table_text, premise, ram_contexts, skip_ram)
            if citation:
                yield "\n" + citation.strip() + "\n\n"

        if not prop_text.endswith((".", "?", "!")):
            prop_text += "."

        if len(prop_text) > MIN_ASSESSABLE_LENGTH:
            citation = await self._assess_and_format(prop_text, premise, ram_contexts, skip_ram)
            prop_text += citation

        yield prop_text + sep

    async def process_chat_message(
        self,
        session_id: str,
        message_text: str,
        skip_guardrails: bool = False,
        attachment_text: Optional[str] = None,
        attachment_filename: Optional[str] = None,
        skip_ivm: Optional[bool] = None,
        skip_ram: Optional[bool] = None,
        skip_nonce: bool = False,
    ) -> AsyncGenerator[str, None]:
        """The main generation pipeline: Safety -> Pre-check -> Context -> Generate -> Assess.

        The three defenses can be disabled independently, which is what lets an
        experiment attribute an effect to one of them rather than to "guardrails
        on vs off" as a single block:

        - ``skip_ivm`` bypasses the IVM safety + relevance checks.
        - ``skip_ram`` bypasses the RAM per-sentence assessment.
        - ``skip_nonce`` bypasses the anti-injection delimiter (see
          ``app.thesis.prompts.build_prompt``), which is a structural defense
          independent of the IVM classifier.

        ``skip_guardrails`` remains as the both-at-once shorthand: it sets
        ``skip_ivm`` and ``skip_ram`` together unless either is passed
        explicitly. It does NOT imply ``skip_nonce`` — the delimiter is not
        part of the IVM/RAM pair, and folding it in would silently change what
        the existing Experiment 4 baseline measures.

        Retrieval always runs so the LLM has context.

        ``attachment_text`` (the extracted text of a chat-uploaded PDF, if
        any) is treated as part of the user's prompt for this turn only: it
        is folded into the same IVM safety check, the same KB search queries,
        and the same anti-injection delimiter as the typed message, but is
        never persisted or replayed on later turns (see
        ``app/chat/infra/pdf_text_extractor.py`` and
        ``app/chat/application/attachment_service.py``).
        """

        # Resolve the individual switches from the shorthand.
        skip_ivm = skip_guardrails if skip_ivm is None else skip_ivm
        skip_ram = skip_guardrails if skip_ram is None else skip_ram

        # Combined text used for safety checking and prompt building: the
        # attachment is data the user is asking about, so it rides inside
        # the same trust boundary as the typed message rather than being
        # treated as separate system-provided context.
        if attachment_text:
            combined_text = (
                f"{message_text}\n\n[Dokumen terlampir: {attachment_filename}]\n{attachment_text}"
            )
            # Capped excerpt for KB search queries only (full text still goes
            # into combined_text above for the IVM check and LLM prompt) —
            # a short question like "is this against X rules?" alone often
            # won't retrieve the right KB chunks since the real topic lives
            # in the attachment, but a full-length document would degrade
            # embedding/HyDE query quality.
            search_query_text = (
                f"{message_text}\n\n{attachment_text[: self.attachment_search_excerpt_chars]}"
            )
        else:
            combined_text = message_text
            search_query_text = message_text

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

        # Record user message. Only the typed text and the attachment's
        # filename are persisted — the extracted attachment text itself is
        # single-turn only and never stored (see docstring above).
        await self.chat_repo.create_message(
            session_id, "user", message_text, raw_content=message_text,
            attachment_filename=attachment_filename,
        )

        try:
            # 2. Safety Check (IVM) — scans the combined message + attachment
            # text, since an attached PDF is just as capable of carrying a
            # prompt injection as typed text.
            if not skip_ivm:
                await self.ivm_service.check_malicious(combined_text)

            # 3. Pre-check Relevance (IVM + KB)
            # NOTE: We intentionally do NOT pass session_id here. The chat
            # session ID is unrelated to the KB chunk session_id payload —
            # KB chunks are ingested without a session_id and the Qdrant
            # filter would return zero results if we passed the chat
            # session ID.
            precheck_contexts = await self.search_service.search(search_query_text, top_k=3)
            if not skip_ivm:
                if not precheck_contexts:
                    raise IrrelevantQueryException("No relevant contexts found in the knowledge base.")

                context_chunks = [ctx.text for ctx in precheck_contexts]
                context_scores = [ctx.score for ctx in precheck_contexts]
                await self.relevance_service.check_relevance(search_query_text, context_chunks, context_scores)

            # 4. Deep Context Retrieval (KB)
            full_contexts = await self.search_service.search(search_query_text, top_k=15)
            
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
            bundle = build_prompt(
                combined_text, ram_contexts, self.system_prompt, use_nonce=not skip_nonce
            )

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
            # Accumulates contiguous Markdown table row propositions until a
            # non-table-row proposition ends the block (see
            # _handle_complete_proposition). Local to this call — never
            # instance state, since ChatService may be reused across
            # concurrent requests.
            table_rows: List[Tuple[str, str]] = []

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
                            async for out in self._handle_complete_proposition(
                                prop, sep, ram_contexts=ram_contexts, premise=premise,
                                skip_ram=skip_ram, table_rows=table_rows,
                            ):
                                final_output += out
                                yield json.dumps({"type": "chunk", "content": out}) + "\n"

                        buffer = propositions[-1][0]

            # Process any remaining buffer through the same row/prose dispatch.
            if buffer.strip():
                for prop, sep in self._split_propositions(buffer.strip()):
                    async for out in self._handle_complete_proposition(
                        prop, sep, ram_contexts=ram_contexts, premise=premise,
                        skip_ram=skip_ram, table_rows=table_rows,
                    ):
                        final_output += out
                        yield json.dumps({"type": "chunk", "content": out}) + "\n"

            # If the answer ended while still inside a table (the last
            # streamed content was table rows, so no trailing prose
            # proposition ever arrived to trigger the flush inside
            # _handle_complete_proposition), assess and flush it now.
            if table_rows:
                table_text = "\n".join(row for row, _ in table_rows)
                table_rows.clear()
                citation = await self._assess_and_format(table_text, premise, ram_contexts, skip_ram)
                if citation:
                    out = "\n" + citation.strip() + "\n\n"
                    final_output += out
                    yield json.dumps({"type": "chunk", "content": out}) + "\n"

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
