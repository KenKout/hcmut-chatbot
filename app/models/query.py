from pydantic import BaseModel
from typing import Optional, List
from haystack import Document

class Query(BaseModel):
    """
    Query model for the API.
    """
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "example query",
            }
        }

class QueryResponse(BaseModel):
    """
    Response model for the API.
    """
    faq_documents: Optional[List[Document]] = None
    web_documents: Optional[List[Document]] = None
    file_documents: Optional[List[Document]] = None
    llm_answer: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "faq_documents": [
                    {
                        "id": "1",
                        "content": "Question",
                        "blob": None,
                        "meta": {
                            "answer": "Answer",
                            "source_id": "0",
                            "page_number": 1,
                            "split_id": 0,
                            "split_idx_start": 0
                        },
                        "score": 0.95,
                        "embedding": None,
                        "sparse_embedding": None
                    }
                ],
                "web_documents": [
                    {
                        "id": "2",
                        "content": "Web content",
                        "blob": None,
                        "meta": {
                            "source_id": "1",
                            "page_number": 1,
                            "split_id": 0,
                            "split_idx_start": 0
                        },
                        "score": 0.90,
                        "embedding": None,
                        "sparse_embedding": None
                    }
                ],
                "file_documents": [
                    {
                        "id": "3",
                        "content": "File content",
                        "blob": None,
                        "meta": {
                            "source_id": "2",
                            "page_number": 1,
                            "split_id": 0,
                            "split_idx_start": 0
                        },
                        "score": 0.85,
                        "embedding": None,
                        "sparse_embedding": None
                    }
                ],
                "llm_answer": "This is the answer from the LLM."
            }
        }