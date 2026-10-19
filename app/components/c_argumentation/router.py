from fastapi import APIRouter

router = APIRouter(tags=["C4 – Multi-Agent Argumentation Engine"])


@router.get("/ping")
async def ping():
    return {"component": "c_argumentation", "status": "ok"}
