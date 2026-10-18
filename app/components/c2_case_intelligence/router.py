from fastapi import APIRouter

router = APIRouter(tags=["C2 – Legal Document Intelligence"])


@router.get("/ping")
async def ping():
    return {"component": "c2", "status": "ok"}
