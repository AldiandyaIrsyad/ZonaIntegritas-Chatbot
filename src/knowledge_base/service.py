import os
import uuid
import aiofiles
from fastapi import UploadFile
from src.knowledge_base.repository import PDFRepository

class KnowledgeBase:
    def __init__(self, repository: PDFRepository):
        self.repository = repository
        self.upload_dir = "upload"
        os.makedirs(self.upload_dir, exist_ok=True)
        
    async def list_pdfs(self):
        pdfs = await self.repository.get_all_pdfs()
        return [{"id": p.id, "title": p.title, "description": p.description, "pdf_path": p.pdf_path, "active": p.active} for p in pdfs]
        
    async def upload_pdf(self, title: str, description: str, file: UploadFile):
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(self.upload_dir, unique_filename)
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        return await self.repository.create_pdf(title, description, file_path)
        
    async def update_pdf_status(self, pdf_id: str, active: bool):
        return await self.repository.update_pdf_active_status(pdf_id, active)
        
    async def delete_pdf(self, pdf_id: str):
        pdf = await self.repository.get_pdf_by_id(pdf_id)
        if pdf and pdf.pdf_path and os.path.exists(pdf.pdf_path):
            os.remove(pdf.pdf_path)
        return await self.repository.delete_pdf(pdf_id)