from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class QueryStats(BaseModel):
    user_query: str = Field(...)
    resolve_time_ms: float = Field(...)
    bot_answer: str = Field(...)
    rag_hit: bool = Field(...)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        collection_name = "query_stats"
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "user_query": "What is the capital of France?",
                "resolve_time_ms": 123.45,
                "bot_answer": "The capital of France is Paris.",
                "rag_hit": True,
                "created_at": "2023-10-27T10:30:00.000Z"
            }
        }
