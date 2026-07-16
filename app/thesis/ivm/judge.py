"""
LLM-based Judge implementation for relevance checking.
"""
import structlog
from app.thesis.ivm.interfaces import IJudge, ILLMJudgeConnection

logger = structlog.get_logger(__name__)


class LLMJudge(IJudge):
    """Judge that uses an LLM to evaluate relevance."""

    def __init__(self, llm_connection: ILLMJudgeConnection, model: str = "llama3-70b-8192") -> None:
        self.llm_connection = llm_connection
        self.model = model
        self.system_prompt = (
            "You are a relevance judge for a retrieval-augmented QA system. Your "
            "task is to determine if a given query is on the same topic or domain "
            "as the provided context, even if the context does not fully or "
            "directly answer it. "
            "Reply with exactly 'YES' if the query relates to the same subject "
            "matter as the context. "
            "Reply with exactly 'NO' only if the query is about a clearly "
            "unrelated topic, or is malicious/nonsensical."
        )

    async def evaluate_relevance(self, query: str, context: str) -> bool:
        """Evaluate relevance using the core LLM connection."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}\n\nIs this relevant?"}
        ]
        
        logger.info("llm_judge.evaluating", query=query[:100])
        try:
            response_chunks = []
            async for chunk in self.llm_connection.stream_chat(
                model=self.model,
                messages=messages,
                max_tokens=50
            ):
                response_chunks.append(chunk)
                
            response_text = "".join(response_chunks).strip().upper()
            logger.info("llm_judge.result", query=query[:100], response=response_text, chunk_count=len(response_chunks))
            
            # Fail closed: if we don't get a clear YES, it's irrelevant.
            return response_text.startswith("YES")
        except Exception as e:
            logger.error("llm_judge.error", error=str(e), exc_info=True)
            # Fail closed on exception
            raise
