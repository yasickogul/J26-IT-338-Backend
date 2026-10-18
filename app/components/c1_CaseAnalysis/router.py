from fastapi import APIRouter

router = APIRouter(tags=["C1 – Case Analysis"])


@router.get("/ping")
async def ping():
    return {"component": "c1", "status": "ok"}
