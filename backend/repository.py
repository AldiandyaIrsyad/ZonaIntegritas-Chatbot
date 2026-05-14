from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from backend.models import Session as DBSession, Message as DBMessage, PDFDocument

class ChatRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_sessions(self):
        result = await self.db.execute(select(DBSession).order_by(DBSession.created_at.asc()))
        return result.scalars().all()

    async def create_session(self, session_id: str, title: str):
        new_session = DBSession(id=session_id, title=title)
        self.db.add(new_session)
        await self.db.commit()
        await self.db.refresh(new_session)
        return new_session

    async def get_session_by_id(self, session_id: str, load_messages: bool = False):
        query = select(DBSession).where(DBSession.id == session_id)
        if load_messages:
            query = query.options(selectinload(DBSession.messages))
        result = await self.db.execute(query)
        return result.scalars().first()

    async def update_session_title(self, session: DBSession, new_title: str):
        session.title = new_title
        await self.db.commit()
        return session

    async def create_message(self, session_id: str, role: str, content: str):
        new_msg = DBMessage(session_id=session_id, role=role, content=content)
        self.db.add(new_msg)
        await self.db.commit()
        return new_msg

    async def delete_session(self, session_id: str):
        session = await self.get_session_by_id(session_id)
        if session:
            await self.db.delete(session)
            await self.db.commit()
            return True
        return False

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
