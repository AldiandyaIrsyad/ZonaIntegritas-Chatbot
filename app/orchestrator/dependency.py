from fastapi import Depends
from app.core.interfaces.ai import IEmbeddingProvider, IReranker
from app.core.interfaces.infra import IVectorStore, IDocumentParser
from app.core.interfaces.ivm import IIVMService
from app.core.interfaces.llm import ILLMService
from app.core.interfaces.rag import IRetrievalService
from app.core.interfaces.ram import IRAMService

from app.chat.dependency import get_session_vector_store
from app.llm import get_llm_service
from app.rag import get_embedding_provider, get_reranker, get_retrieval_service, get_document_parser
from app.ivm.dependency import get_ivm_service
from app.ram.dependency import get_ram_service

from app.orchestrator.service import ChatOrchestrator

def get_chat_orchestrator(
    llm_service: ILLMService = Depends(get_llm_service),
    retrieval_service: IRetrievalService = Depends(get_retrieval_service),
    reranker: IReranker = Depends(get_reranker),
    vector_store: IVectorStore = Depends(get_session_vector_store),
    embedding_provider: IEmbeddingProvider = Depends(get_embedding_provider),
    ivm_service: IIVMService = Depends(get_ivm_service),
    ram_service: IRAMService = Depends(get_ram_service),
    document_parser: IDocumentParser = Depends(get_document_parser),
) -> ChatOrchestrator:
    return ChatOrchestrator(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        reranker=reranker,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        ivm_service=ivm_service,
        ram_service=ram_service,
        document_parser=document_parser,
    )
