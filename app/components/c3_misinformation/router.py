from fastapi import APIRouter

from .api import router as claim_router

router = APIRouter(tags=["C3 – Misinformation Detection"])


@router.get("/ping")
async def ping():
    return {"component": "c3", "status": "ok"}


router.include_router(claim_router)
