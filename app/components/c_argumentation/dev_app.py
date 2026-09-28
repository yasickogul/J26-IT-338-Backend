# Run only this component:
#   uvicorn app.components.c_argumentation.dev_app:app --reload --port 8004
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

from .router import router

app = FastAPI(title="C4 – Multi-Agent Argumentation Engine (dev)")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1/c_argumentation")
