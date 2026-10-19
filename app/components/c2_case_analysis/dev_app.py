"""Run from the backend repository root on port 8000."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .router import router
from .runtime import lifespan

app = FastAPI(title="C2 Civil Case Intelligence API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1/c2")
