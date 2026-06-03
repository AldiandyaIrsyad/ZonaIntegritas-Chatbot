"""JSON API endpoints for RAM."""
from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel
import structlog

from app.core.interfaces.ram import IRAMService
from .dependency import get_ram_service

logger = structlog.get_logger(__name__)

router = APIRouter()

class AssessRequest(BaseModel):
    """Payload for assessing a sentence against a premise."""
    premise: str
    sentence: str

@router.post("/api/ram/assess")
async def assess_sentence(
    req: AssessRequest,
    service: IRAMService = Depends(get_ram_service),
) -> Dict[str, Any]:
    """Test RAM sentence assessment manually."""
    logger.info("Assessing sentence manually via API", premise_len=len(req.premise), sentence_len=len(req.sentence))
    # Note: We pass empty contexts since this is just a raw test endpoint
    result = await service.assess_sentence(
        sentence=req.sentence,
        premise=req.premise,
        contexts=[],
        context_embs=None,
    )
    return {
        "label": result.label,
        "entailment_score": result.entailment_score,
        "contradiction_score": result.contradiction_score,
        "neutral_score": result.neutral_score,
    }
