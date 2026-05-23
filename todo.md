[X] Make sure the list is reverse
[X] CRUD for the chat. Make sure you can delete
[X] Upload PDF passthrough to upload
[x] Refactor frontend to use Alpine.js
[x] Refactor packages. use the init to run the initial setup so main is more lenient.
[x] Create a new python folder package for LLMs.
[x] Refactor LLMs package, and service.py to streamlined the use of getenv for model usage. Create LLM_MODE, LLM_API, and LLM_Model env to be used.
[x] Create services folder and use that instead of LLMs, this service layer will contain IVM, RAG, PDF parser, ETC.
[x] Make sure chat can read previous message as context use langchain and tiktoken for this
[x] Make sure there is severals roles for chat. System, User, AI, which we will use salted later.
[x] Add system prompt mechanism for `/admin` and into the chat.

## Reminder
REWRITE Uploads. Make sure there is admin upload (I.e. knowledge base upload) and user upload


## MAJOR FEAT RAG

[ ] Implement docker for vector database (Qdrant)
[ ] Refactor docker-compose use profile to make it's only ran necessary things.
[ ] `src/infra/vector_store.py` A dedicated abstraction layer for the vector database client. Handles connection pooling, index creation, raw upserts, and payload-filtered similarity searches.
[ ] `src/rag/ingestion.py` The write path. Handles the asynchronous ingestion pipeline (Admin Dashboard -> PDF parsing -> Hierarchical Chunking -> Embedding -> vector_store.upsert).
[ ] `src/rag/retrieval.py` The read path. Orchestrates the Contextualization Engine (Query -> Query Decomposition -> Hybrid Retrieval via vector_store -> Cross-Encoder Re-ranking -> Top-K Context).

### Notes

Prioritize Qdrant over ChromaDB or FAISS. Qdrant natively supports hybrid search (Dense Vectors + Sparse BM25 vectors) within a single query execution. It also excels at payload (metadata) filtering, which is critical for your state management requirements.

Parsing & Chunking: unstructured + PyMuPDF
Standard token splitters destroy the semantic layout of formal documents like university regulations. Use the unstructured Python library. It uses layout-aware models (e.g., YOLOX) to identify titles, sections, tables, and narrative text, enabling true hierarchical semantic chunking before passing the text to an embedding model.

Embedding Model: BAAI/bge-m3
This model is highly optimized for multilingual contexts, including Bahasa Indonesia. The "M3" stands for Multi-lingual, Multi-granularity, and Multi-Representation. Crucially, it simultaneously outputs dense vectors (for semantic search) and sparse lexical weights (for BM25 exact-keyword matching), fitting perfectly into the hybrid retrieval node in your diagram.

Re-Ranker: BAAI/bge-reranker-v2-m3
Deploy this cross-encoder model to re-score the raw chunks retrieved by Qdrant. It computes the attention between the user query and the retrieved document chunk directly, maximizing context precision before feeding data to the Prompt Builder.

Managing PDF State (Enable/Disable)

Do not execute computationally expensive database rebuilds or hard deletions to toggle PDF availability. Manage document states exclusively through metadata filtering.

    Ingestion: When embedding chunks, attach a dictionary payload: {"doc_id": "sk_rektor_2024", "category": "akademik", "is_active": true}.

    Disabling: To disable a PDF, issue a fast update operation to the vector database to toggle is_active: false for all vectors matching "doc_id": "sk_rektor_2024".

    Retrieval: Hardcode a payload filter into the retrieval.py query logic so the engine only searches vectors where is_active == true.

## QoL compare to other LLMs chat.

## Input Validation Module

[ ] Create PDF parser for chat, rename the current service into chat PDF and also KB (knowledge base) PDF.
[ ] Create IVM (Input Validation module)
[ ] Make sure when LLMs answering a question it give "source" and PDFs as source.
[ ] Initialize vector database for RAG
[ ] Refactor service.py PDFParser to it's own, focusing on inserting to database, and chat input
[ ] Make user able to upload PDFs.
