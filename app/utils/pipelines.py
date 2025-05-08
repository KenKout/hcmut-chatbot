from app.envs import settings
from app.database import database
from haystack import Pipeline
from haystack.components.embedders import (
    OpenAITextEmbedder,
    HuggingFaceAPITextEmbedder,
    SentenceTransformersTextEmbedder
)
from app.utils.embedders import embedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.utils import Secret
from pathlib import Path
from typing import List, Dict, Any

from app.models.upload import FileUploadParams
from haystack.components.converters import MultiFileConverter
from haystack.components.preprocessors import DocumentPreprocessor
from haystack.components.writers import DocumentWriter


class ChatPipeline:
    def __init__(self):
        self.pipeline = Pipeline()
        self.faq_store = database.faq_documents_store
        self.web_store = database.web_documents_store
        self.file_store = database.file_documents_store

        if settings.EMBEDDING_PROVIDER == "huggingface":
            self.query_embedder = HuggingFaceAPITextEmbedder(
                api_type=settings.EMBEDDING_HUGGINGFACE_API_TYPE,
                api_params={"url": settings.EMBEDDING_HUGGINGFACE_BASE_URL} if settings.EMBEDDING_HUGGINGFACE_API_TYPE != "serverless_inference_api" else {"model": settings.EMBEDDING_MODEL},
                token=Secret.from_token(settings.EMBEDDING_HUGGINGFACE_API_KEY),
            )
        elif settings.EMBEDDING_PROVIDER == "openai":
            self.query_embedder = OpenAITextEmbedder(
                api_key=Secret.from_token(settings.EMBEDDING_OPENAI_API_KEY),
                api_base_url=settings.EMBEDDING_OPENAI_BASE_URL,
                model=settings.EMBEDDING_MODEL,
                dimensions=settings.EMBEDDING_DIM,
            )
        elif settings.EMBEDDING_PROVIDER == "sentence_transformers":
            self.query_embedder = SentenceTransformersTextEmbedder(
                model=settings.EMBEDDING_MODEL,
                token=Secret.from_token(settings.EMBEDDING_HUGGINGFACE_API_KEY),
            )
        else:
            raise ValueError(f"Unsupported embedding provider for query: {settings.EMBEDDING_PROVIDER}")

        # 2. Initialize Retrievers
        self.faq_retriever = QdrantEmbeddingRetriever(
            document_store=self.faq_store,
            top_k=settings.EMBEDDING_TOP_K,
            score_threshold=settings.EMBEDDING_THRESHOLD
        )
        self.web_retriever = QdrantEmbeddingRetriever(
            document_store=self.web_store,
            top_k=settings.EMBEDDING_TOP_K,
            score_threshold=settings.EMBEDDING_THRESHOLD
        )
        self.file_retriever = QdrantEmbeddingRetriever(
            document_store=self.file_store,
            top_k=settings.EMBEDDING_TOP_K,
            score_threshold=settings.EMBEDDING_THRESHOLD
        )

        # 3. Build the retrieval pipeline
        self.pipeline.add_component("query_embedder", self.query_embedder)
        self.pipeline.add_component("faq_retriever", self.faq_retriever)
        self.pipeline.add_component("web_retriever", self.web_retriever)
        self.pipeline.add_component("file_retriever", self.file_retriever)

        # Connect components
        self.pipeline.connect("query_embedder.embedding", "faq_retriever.query_embedding")
        self.pipeline.connect("query_embedder.embedding", "web_retriever.query_embedding")
        self.pipeline.connect("query_embedder.embedding", "file_retriever.query_embedding")

    def run(self, query: str):
        pipeline_output = self.pipeline.run({
            "query_embedder": {"text": query}
        })
        return {
            "faq_documents": pipeline_output["faq_retriever"]["documents"],
            "web_documents": pipeline_output["web_retriever"]["documents"],
            "file_documents": pipeline_output["file_retriever"]["documents"]
        }


class FileProcessingPipeline:
    def __init__(self, file_upload_params: FileUploadParams):
        self.pipeline = Pipeline()
        self.file_store = database.file_documents_store

        # 1. Define Converter
        self.converter = MultiFileConverter()
        self.pipeline.add_component("converter", self.converter)

        # 2. DocumentPreprocessor
        self.preprocessor = DocumentPreprocessor(
            split_by=file_upload_params.split_by,
            split_length=file_upload_params.split_length,
            split_overlap=file_upload_params.split_overlap,
            remove_empty_lines=file_upload_params.remove_empty_lines,
            remove_extra_whitespaces=file_upload_params.remove_extra_whitespaces,
        )
        self.pipeline.add_component("preprocessor", self.preprocessor)

        # 5. Embedder (using the global embedder instance from app.utils.embedders)
        self.pipeline.add_component("embedder", embedder)

        # 6. DocumentWriter
        self.document_writer = DocumentWriter(document_store=self.file_store, policy="OVERWRITE")
        self.pipeline.add_component("document_writer", self.document_writer)

        # Connect components:
        # Converter -> Preprocessor
        self.pipeline.connect("converter.documents", "preprocessor.documents")
        # Preprocessor -> Embedder
        self.pipeline.connect("preprocessor.documents", "embedder.documents")
        # Embedder -> DocumentWriter
        self.pipeline.connect("embedder.documents", "document_writer.documents")

    def run(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Runs the file processing pipeline.
        :param file_paths: A list of pathlib.Path objects pointing to the files to process.
        :return: The output of the Haystack pipeline run.
        """
        return self.pipeline.run(data={"converter": {"sources": file_paths}})
