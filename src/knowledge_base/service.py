import os
from fastapi import UploadFile
from src.knowledge_base.repository import PDFRepository
from src.infra.storage import StorageProvider


class KnowledgeBase:
    def __init__(self, repository: PDFRepository, storage: StorageProvider):
        self.repository = repository
        self.storage = storage
        
    async def list_pdfs(self):
        pdfs = await self.repository.get_all_pdfs()
        return [{"id": p.id, "title": p.title, "description": p.description, "pdf_path": p.pdf_path, "active": p.active} for p in pdfs]
        
    async def upload_pdf(self, title: str, description: str, file: UploadFile):
        file_extension = os.path.splitext(file.filename or "")[1].lower()
        if file.content_type != "application/pdf" or file_extension != ".pdf":
            raise ValueError("Only PDF files are allowed")
            
        file_path = await self.storage.save_file(file, file_extension)
            
        return await self.repository.create_pdf(title, description, file_path)
        
    async def update_pdf_status(self, pdf_id: str, active: bool):
        return await self.repository.update_pdf_active_status(pdf_id, active)
        
    async def delete_pdf(self, pdf_id: str):
        pdf = await self.repository.get_pdf_by_id(pdf_id)
        if pdf and pdf.pdf_path:
            await self.storage.delete_file(pdf.pdf_path)
        return await self.repository.delete_pdf(pdf_id)