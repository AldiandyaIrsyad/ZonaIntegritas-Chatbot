from typing import Protocol, List, Any, Optional
from pydantic import BaseModel

class ChunkData(BaseModel):
    id: str
    text: str
    chunk_index: int
    page: Optional[int]

class ISessionChunkProvider(Protocol):
    async def get_session_chunks_by_ids(self, chunk_ids: List[str]) -> List[Any]:
        ...
        
    async def save_document_chunks(self, doc_id: str, chunks: List[ChunkData]) -> None:
        ...
