# Run only this component:
#   uvicorn app.components.c3_misinformation.dev_app:app --reload --port 8003
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

from .router import router

app = FastAPI(title="C3 – Misinformation Detection (dev)")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1/c3")
