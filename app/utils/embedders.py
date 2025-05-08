from app.envs import settings
from haystack.utils import Secret
from haystack.components.embedders import OpenAIDocumentEmbedder, HuggingFaceAPIDocumentEmbedder, SentenceTransformersDocumentEmbedder

class Embedders:
    def __init__(self):
        if settings.EMBEDDING_PROVIDER == "huggingface":
            self.embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=settings.EMBEDDING_HUGGINGFACE_API_TYPE,
                api_params={"url": settings.EMBEDDING_HUGGINGFACE_BASE_URL} if settings.EMBEDDING_HUGGINGFACE_API_TYPE != "serverless_inference_api" else {"model": settings.EMBEDDING_MODEL},
                token=Secret.from_token(settings.EMBEDDING_HUGGINGFACE_API_KEY),
                batch_size=settings.DB_BATCH_SIZE,
            )
        elif settings.EMBEDDING_PROVIDER == "openai":
            self.embedder = OpenAIDocumentEmbedder(
                api_key=Secret.from_token(settings.EMBEDDING_OPENAI_API_KEY),
                api_base_url=settings.EMBEDDING_OPENAI_BASE_URL,
                model=settings.EMBEDDING_MODEL,
                batch_size=settings.DB_BATCH_SIZE,
                dimensions=settings.EMBEDDING_DIM,
            )
        elif settings.EMBEDDING_PROVIDER == "sentence_transformers":
            self.embedder = SentenceTransformersDocumentEmbedder(
                model=settings.EMBEDDING_MODEL,
                token=Secret.from_token(settings.EMBEDDING_HUGGINGFACE_API_KEY),
            )
            self.embedder.warm_up()
        else:
            raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")

embedder = Embedders().embedder