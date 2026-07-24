"""In-process BGE-M3 dense+sparse embedding adapter.

Runs BGE-M3 in-process rather than over HTTP via Infinity (the other
embedding/reranking server this app uses, see ``infinity_embeddings.py``):
Infinity's own model list documents ``BAAI/bge-m3`` as dense-only ("no
sparse") — this is a server limitation, not a misconfiguration (confirmed
against open, unresolved upstream GitHub issues on the Infinity project).
This adapter replaces Infinity for embeddings specifically, using BAAI's own
reference implementation (``FlagEmbedding.BGEM3FlagModel``), which is the
only path that actually computes BGE-M3's lexical (sparse) weights alongside
the dense vector. See ``docs/02-arsitektur.md`` §2.1 for the broader
architecture context.

Fulfills: ``app/kb/domain/interfaces.py::ITextEmbedder``.
Wired in: ``app/kb/dependency.py::get_text_embedder``.
"""

import asyncio
from functools import lru_cache
from typing import List

import structlog
import torch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.kb.domain.interfaces import ITextEmbedder, EmbeddingResult

logger = structlog.get_logger(__name__)

# Retries transient CUDA OOM (e.g. a momentary VRAM spike from another
# process sharing the GPU) — not a network error, so this doesn't reuse
# app.shared.retry.external_api_retry, which targets httpx exceptions.
# Clears the CUDA cache before each retry attempt since simply retrying
# without freeing anything would likely OOM again.
def _clear_cuda_cache_before_retry(retry_state) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_cuda_oom_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(torch.cuda.OutOfMemoryError),
    before_sleep=_clear_cuda_cache_before_retry,
    reraise=True,
)


@lru_cache(maxsize=1)
def _load_model(model_name: str, use_fp16: bool, device: str):
    """Load BGEM3FlagModel once as a process-lifetime singleton.

    Model load takes several seconds and several GB of VRAM/RAM, so this
    must not be repeated per-request — every BGEM3Embeddings instance
    sharing the same (model_name, use_fp16, device) reuses this one model.
    """
    from FlagEmbedding import BGEM3FlagModel

    logger.info("bge_m3.loading", model=model_name, device=device, use_fp16=use_fp16)
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
    logger.info("bge_m3.loaded", model=model_name, device=device)
    return model


class BGEM3Embeddings(ITextEmbedder):
    """Dense + sparse text embedding adapter backed by an in-process BGE-M3.

    Fulfills: ``app/kb/domain/interfaces.py::ITextEmbedder``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        use_fp16: bool = True,
        batch_size: int = 12,
    ) -> None:
        """Store model config; the model itself is loaded lazily (and once
        per process) by :func:`_load_model` on first ``embed_texts`` call."""
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        logger.info(
            "BGEM3Embeddings initialized",
            model=model_name,
            device=device,
            use_fp16=use_fp16,
        )

    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Encode texts via the shared BGE-M3 model, returning dense vectors
        and lexical (sparse) weights for each. Retries once-transient CUDA
        OOM through :data:`_cuda_oom_retry` on the inner :meth:`_encode` call."""
        if not texts:
            return []

        try:
            output = await self._encode(texts)
        except Exception as exc:
            logger.error("bge_m3.embed_failed", batch_size=len(texts), error=str(exc))
            raise

        results: List[EmbeddingResult] = []
        for dense_vec, lexical_weights in zip(output["dense_vecs"], output["lexical_weights"]):
            sparse_indices = [int(k) for k in lexical_weights.keys()]
            sparse_values = [float(v) for v in lexical_weights.values()]
            results.append(
                EmbeddingResult(
                    dense=dense_vec.tolist(),
                    sparse_indices=sparse_indices,
                    sparse_values=sparse_values,
                )
            )
        return results

    @_cuda_oom_retry
    async def _encode(self, texts: List[str]) -> dict:
        model = _load_model(self.model_name, self.use_fp16, self.device)
        return await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    async def close(self) -> None:
        # The model is a process-lifetime singleton shared across instances —
        # nothing to release per-instance.
        pass
