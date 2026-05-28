from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from .model import PDFDocument


class PDFRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
        
    async def get_all_pdfs(self):
        result = await self.db.execute(select(PDFDocument).order_by(PDFDocument.created_at.desc()))
        return result.scalars().all()
        
    async def get_pdf_by_id(self, pdf_id: str):
        query = select(PDFDocument).where(PDFDocument.id == pdf_id)
        result = await self.db.execute(query)
        return result.scalars().first()
        
    async def create_pdf(self, title: str, description: str, pdf_path: str):
        new_pdf = PDFDocument(title=title, description=description, pdf_path=pdf_path)
        self.db.add(new_pdf)
        await self.db.commit()
        await self.db.refresh(new_pdf)
        return new_pdf
        
    async def update_pdf_active_status(self, pdf_id: str, active: bool):
        pdf = await self.get_pdf_by_id(pdf_id)
        if pdf:
            pdf.active = active
            await self.db.commit()
            await self.db.refresh(pdf)
            return pdf
        return None
        
    async def delete_pdf(self, pdf_id: str):
        pdf = await self.get_pdf_by_id(pdf_id)
        if pdf:
            await self.db.delete(pdf)
            await self.db.commit()
            return True
        return False
