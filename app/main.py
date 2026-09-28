from contextlib import asynccontextmanager
from importlib import import_module
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import check_connection
from app.registry import COMPONENTS

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    result = await check_connection()
    if result.get("ok"):
        msg = f"Database connected ({result.get('database')})"
        print(f"✅ {msg}", flush=True)
        logger.info(msg)
    else:
        msg = f"Database connection failed: {result.get('error')}"
        print(f"❌ {msg}", flush=True)
        logger.error(msg)
    yield


app = FastAPI(
    title="Smart Civil Case Analysis Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


@app.get("/health/db", tags=["system"])
async def health_db():
    result = await check_connection()
    if not result.get("ok"):
        return JSONResponse(status_code=503, content={"status": "error", **result})
    return {"status": "ok", **result}


for module_path, prefix in COMPONENTS.items():
    try:
        module = import_module(f"{module_path}.router")
        app.include_router(module.router, prefix=f"/api/v1/{prefix}")
    except Exception as e:  # a broken component must not crash the others
        print(f"[WARN] component {module_path} not loaded: {e}")
