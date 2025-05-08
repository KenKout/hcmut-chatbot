import pandas as pd
from loguru import logger
from tqdm import tqdm
from app.envs import settings
from app.utils.embedders import embedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack import Document
from haystack.components.preprocessors import DocumentPreprocessor
from haystack.utils import Secret

class Database:
    def __init__(self):
        if settings.DEBUG:
            self.faq_documents_store = QdrantDocumentStore(
                url=":memory:",
                embedding_dim=settings.EMBEDDING_DIM,
            )

            self.web_documents_store = QdrantDocumentStore(
                url=":memory:",
                embedding_dim=settings.EMBEDDING_DIM,
            )
            self.file_documents_store = QdrantDocumentStore( # New store for general files
                url=":memory:",
                embedding_dim=settings.EMBEDDING_DIM,
            )
        else:
            self.faq_documents_store = QdrantDocumentStore(
                url=settings.QDRANTDB_URL,
                api_key=Secret.from_token(settings.QDRANTDB_API_KEY),
                embedding_dim=settings.EMBEDDING_DIM,
                hnsw_config={"m": 128},
                timeout=settings.DB_TIMEOUT,
                write_batch_size=settings.DB_BATCH_SIZE,
                recreate_index=False,
                index="faq",
            )

            self.web_documents_store = QdrantDocumentStore(
                url=settings.QDRANTDB_URL,
                api_key=Secret.from_token(settings.QDRANTDB_API_KEY),
                embedding_dim=settings.EMBEDDING_DIM,
                hnsw_config={"m": 128},
                timeout=settings.DB_TIMEOUT,
                write_batch_size=settings.DB_BATCH_SIZE,
                recreate_index=False,
                index="web",
            )

            self.file_documents_store = QdrantDocumentStore(
                url=settings.QDRANTDB_URL,
                api_key=Secret.from_token(settings.QDRANTDB_API_KEY),
                embedding_dim=settings.EMBEDDING_DIM,
                hnsw_config={"m": 128},
                timeout=settings.DB_TIMEOUT,
                write_batch_size=settings.DB_BATCH_SIZE,
                recreate_index=False,
                index="files",
            )
    
    def reindex(self, faq_file, web_file, dev=False, batch_size=None):
        """
        Reindex documents from FAQ and web data files.
        
        Args:
            faq_file: Path to the FAQ file (CSV or JSON)
            web_file: Path to the web data file (CSV or JSON)
            dev: If True, only a small subset of data will be indexed
            batch_size: Batch size for writing documents to the document store
        """
        # Create preprocessor
        processor = DocumentPreprocessor(
            split_by="passage",
            split_length=1,
            split_overlap=0,
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=True,
            respect_sentence_boundary=False,
        )
        
        # Use the provided batch size or the default from settings
        db_batch_size = batch_size or settings.DB_BATCH_SIZE
        
        # Process FAQ data
        if faq_file.endswith(".csv"):
            faq_df = pd.read_csv(faq_file)
        elif faq_file.endswith(".json"):
            faq_df = pd.read_json(faq_file)
        else:
            raise ValueError("FAQ file must be either CSV or JSON")

        if not ("query" in faq_df.columns and "answer" in faq_df.columns):
            raise KeyError("FAQ file must have two keys 'query' and 'answer'")

        # Process web data
        if web_file.endswith(".csv"):
            web_df = pd.read_csv(web_file)
        elif web_file.endswith(".json"):
            web_df = pd.read_json(web_file)
        else:
            raise ValueError("Web file must be either CSV or JSON")

        if not ("text" in web_df.columns and "tables" in web_df.columns):
            raise KeyError("WEB file must have two keys 'text' and 'tables'")

        # Use smaller subset for development
        if dev:
            faq_df = faq_df.head(5)
            web_df = web_df.head(5)

        # Process FAQ documents
        faq_documents = []
        idx = 0
        for _, d in tqdm(faq_df.iterrows(), desc="Loading FAQ..."):
            content = d["query"]
            faq_documents.append(
                Document(content=content, id=str(idx), meta={"answer": d["answer"]})
            )
            idx += 1

        # Process documents with the preprocessor
        processed_faq = processor.run(documents=faq_documents)
        embedder.run(processed_faq["documents"])
        self.faq_documents_store.write_documents(
            documents=processed_faq["documents"],
            policy="OVERWRITE",
        )
        # Process web documents
        web_documents = []
        idx = 0
        for _, d in tqdm(web_df.iterrows(), desc="Loading web data..."):
            content = d["text"]
            web_documents.append(Document(content=content, id=str(idx)))
            idx += 1

            if len(d["tables"]) > 0:
                for table in d["tables"]:
                    web_documents.append(
                        Document(content=table, content_type="table", id=str(idx))
                    )
                    idx += 1

        # Process documents with the preprocessor
        processed_web = processor.run(documents=web_documents)
        embedder.run(processed_web["documents"])
        self.web_documents_store.write_documents(
            documents=processed_web["documents"],
            policy="OVERWRITE",
        )


database = Database()
