import os
from loguru import logger
from enum import Enum
from app.models.embedding import EmbeddingProvider
from dotenv import load_dotenv
from haystack.utils.hf import HFEmbeddingAPIType

class Settings:
    def __init__(self):
        load_dotenv(override=True)

        # Development Settings
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # FastAPI Settings
        self.APP_NAME = os.getenv("APP_NAME", "FastAPI")
        self.APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
        self.APP_DESCRIPTION = os.getenv("APP_DESCRIPTION", "FastAPI")
        self.APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
        self.APP_PORT = int(os.getenv("APP_PORT", 8000))
        self.APP_RELOAD = os.getenv("APP_RELOAD", "false").lower() == "true"
        self.APP_CORS_ORIGINS = os.getenv("APP_CORS_ORIGINS", "*")
        self.APP_WORKERS = int(os.getenv("APP_WORKERS", 1))

        # Logging Settings
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.LOG_FILE = os.getenv("LOG_FILE", None)

        # Database Settings
        self.QDRANTDB_URL = os.getenv("QDRANTDB_URL", "http://localhost:6333")
        self.QDRANTDB_API_KEY = os.getenv("QDRANTDB_API_KEY", "YOUR_QDRANT_API_KEY")
        self.DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", 256))
        self.DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", 60))

        # Embedding Settings
        self.EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", EmbeddingProvider.OPENAI.value)
        if self.EMBEDDING_PROVIDER not in [provider.value for provider in EmbeddingProvider]:
            raise ValueError(f"Invalid EMBEDDING_PROVIDER: {self.EMBEDDING_PROVIDER}. Must be one of {[provider.value for provider in EmbeddingProvider]}")
        
        self.EMBEDDING_HUGGINGFACE_API_KEY = os.getenv("EMBEDDING_HUGGINGFACE_API_KEY", "")
        self.EMBEDDING_HUGGINGFACE_BASE_URL = os.getenv("EMBEDDING_HUGGINGFACE_BASE_URL", "https://api-inference.huggingface.co")
        self.EMBEDDING_HUGGINGFACE_API_TYPE = os.getenv("EMBEDDING_HUGGINGFACE_API_TYPE", HFEmbeddingAPIType.INFERENCE_ENDPOINTS.value)
        self.EMBEDDING_OPENAI_API_KEY = os.getenv("EMBEDDING_OPENAI_API_KEY", "")
        self.EMBEDDING_OPENAI_BASE_URL = os.getenv("EMBEDDING_OPENAI_BASE_URL", "https://api.openai.com/v1")

        # Embedding Configuration
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bkai-foundation-models/vietnamese-bi-encoder")
        self.EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))
        self.EMBEDDING_TOP_K = int(os.getenv("EMBEDDING_TOP_K", 3))
        self.EMBEDDING_THRESHOLD = float(os.getenv("EMBEDDING_THRESHOLD", 0.7))

        # LLM Settings
        self.LLM_OPENAI_API_KEY = os.getenv("LLM_OPENAI_API_KEY", "")
        self.LLM_OPENAI_BASE_URL = os.getenv("LLM_OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.LLM_OPENAI_MODEL = os.getenv("LLM_OPENAI_MODEL", "gpt-3.5-turbo")
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
        self.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1000))

        # Cache Settings
        self.CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() == "true"
        self.FAQ_ENABLE_PARAPHRASING = os.getenv("FAQ_ENABLE_PARAPHRASING", "false").lower() == "true"


settings = Settings()
