from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from fastapi import UploadFile

class FileUploadParams(BaseModel):
    split_by: Literal["word", "page", "sentence", "passage", "function", "period", "line"] = Field(
        default="word",
        description="Unit by which to split the documents. Options: 'word', 'page', 'sentence', 'passage', 'function', 'period', 'line'."
    )
    split_length: int = Field(
        default=200,  # Adjusted default based on common usage
        description="The maximum number of units in each split."
    )
    split_overlap: int = Field(
        default=0,
        description="The number of units that each split overlaps with the previous one."
    )
    remove_empty_lines: bool = Field(
        default=True,
        description="Whether to remove empty lines from the documents."
    )
    remove_extra_whitespaces: bool = Field(
        default=True,
        description="Whether to remove extra whitespaces from the documents."
    )
    # respect_sentence_boundary: bool = Field( # Example of another param
    #     default=True,
    #     description="Whether to respect sentence boundaries when splitting."
    # )

class FileUploadRequest(BaseModel):
    params: FileUploadParams = Field(default_factory=FileUploadParams)
    # We'll handle files separately in the endpoint using FastAPI's UploadFile
    # files: List[UploadFile] # This would be for when files are part of the JSON body, not typical for multipart/form-data

    class Config:
        json_schema_extra = {
            "example": {
                "params": {
                    "split_by": "sentence",
                    "split_length": 10,
                    "split_overlap": 1
                }
            }
        }