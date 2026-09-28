from fastapi import APIRouter

router = APIRouter(tags=["C1 – Document Understanding"])


@router.get("/ping")
async def ping():
    return {"component": "c1", "status": "ok"}
