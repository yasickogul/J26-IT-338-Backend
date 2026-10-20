"""Central configuration. Everything the notebook hard-coded or read from Kaggle Secrets
now comes from environment variables (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Secrets
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "law-rag-kb")

# LLM
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-6-luna")  # confirm exact slug on openrouter.ai/models

# Retrieval (notebook cell 19)
RETRIEVAL_TOP_K_CASES = int(os.getenv("RETRIEVAL_TOP_K_CASES", "3"))
RETRIEVAL_TOP_K_LAW = int(os.getenv("RETRIEVAL_TOP_K_LAW", "5"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))  # cosine similarity

# Web server
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tiff", "bmp"}  # notebook cell 5

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FEWSHOT_CACHE = DATA_DIR / "fewshot.json"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
