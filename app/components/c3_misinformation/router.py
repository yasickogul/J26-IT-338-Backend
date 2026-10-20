from fastapi import APIRouter

from .api import router as claim_router
from .image_api import router as image_claim_router

router = APIRouter(tags=["C3 – Misinformation Detection"])


@router.get("/ping")
async def ping():
    return {"component": "c3", "status": "ok"}


router.include_router(claim_router)
router.include_router(image_claim_router)
