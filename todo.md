[X] Make sure the list is reverse
[X] CRUD for the chat. Make sure you can delete
[X] Upload PDF passthrough to upload
[x] Refactor frontend to use Alpine.js
[x] Refactor packages. use the init to run the initial setup so main is more lenient.
[x] Create a new python folder package for LLMs.
[x] Refactor LLMs package, and service.py to streamlined the use of getenv for model usage. Create LLM_MODE, LLM_API, and LLM_Model env to be used.
[x] Create services folder and use that instead of LLMs, this service layer will contain IVM, RAG, PDF parser, ETC.
[x] Make sure chat can read previous message as context use langchain and tiktoken for this
[ ] Make sure there is severals roles for chat. System, User, AI, which we will use salted later.
[ ] Add system prompt mechanism for `/admin` and into the chat.
[ ] Create PDF parser for chat, rename the current service into chat PDF and also KB (knowledge base) PDF.
[ ] Create IVM (Input Validation module)
[ ] Make sure when LLMs answering a question it give "source" and PDFs as source.
[ ] Initialize vector database for RAG
[ ] Refactor service.py PDFParser to it's own, focusing on inserting to database, and chat input
[ ] Make user able to upload PDFs.
