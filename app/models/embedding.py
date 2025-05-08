from enum import Enum

class EmbeddingProvider(str, Enum):
    """
    Enum for different embedding models.
    """
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCETRANSFORMERS = "sentence_transformers"
