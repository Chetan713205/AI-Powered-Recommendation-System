import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)


class Config:
    OPENROUTER_API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    HF_TOKEN = os.getenv("HF_TOKEN")
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_MODEL = "openrouter/free"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1" 