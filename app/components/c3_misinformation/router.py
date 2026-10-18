from fastapi import APIRouter

router = APIRouter(tags=["C3 – Misinformation Detection"])


@router.get("/ping")
async def ping():
    return {"component": "c3", "status": "ok"}
