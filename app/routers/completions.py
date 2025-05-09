import json
import time
import uuid
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool

from app.utils.pipelines import ChatPipeline
from app.utils.llm import LLM
from app.envs import settings
from loguru import logger

def _extract_source_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    if not meta or not isinstance(meta, dict):
        return None
    source_keys = ['name', 'filename', 'file_path', 'url', 'link', 'source_id', 'title']
    for key in source_keys:
        if key in meta and isinstance(meta[key], str):
            return meta[key]
    if 'id' in meta and isinstance(meta['id'], str):
        return meta['id']
    return None

def _format_documents_for_citation(haystack_docs: List[Any]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    if not haystack_docs:
        return citations
    for doc in haystack_docs:        
        doc_meta = getattr(doc, 'meta', {})
        if not isinstance(doc_meta, dict):
            doc_meta = {}

        citations.append({
            "id": str(getattr(doc, 'id', uuid.uuid4().hex)),
            "content": doc.content,
            "source": _extract_source_from_meta(doc_meta),
            "score": getattr(doc, 'score', None),
            "meta": doc_meta
        })
    return citations

router = APIRouter()

async def _get_documents_and_context(query: str) -> tuple[List[Dict[str, Any]], str]:
    chat_pipeline = ChatPipeline()
    try:
        pipeline_output = await run_in_threadpool(chat_pipeline.run, query)
    except Exception as e:
        logger.error(f"ChatPipeline run error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

    faq_documents = pipeline_output.get("faq_documents", [])
    web_documents = pipeline_output.get("web_documents", [])
    file_documents = pipeline_output.get("file_documents", [])

    all_retrieved_docs = (faq_documents or []) + (web_documents or []) + (file_documents or [])
    
    try:
        all_retrieved_docs.sort(key=lambda x: getattr(x, 'score', 0) or 0, reverse=True)
    except Exception as e:
        logger.warning(f"Could not sort documents by score: {e}")

    citations = _format_documents_for_citation(all_retrieved_docs)

    context_parts = []
    if faq_documents:
        context_parts.extend([doc.content for doc in faq_documents if doc.content])
    if web_documents:
        context_parts.extend([doc.content for doc in web_documents if doc.content])
    if file_documents:
        context_parts.extend([doc.content for doc in file_documents if doc.content])
    
    context = "\n\n".join(context_parts)
    return citations, context

async def _stream_response_generator(
    llm_client_stream: AsyncGenerator, 
    citations_list: List[Dict[str, Any]]
) -> AsyncGenerator[str, None]:
    
    async for chunk in llm_client_stream:
        delta_content = chunk.choices[0].delta.content
        delta_role = chunk.choices[0].delta.role
        finish_reason = chunk.choices[0].finish_reason

        current_delta: Dict[str, Any] = {}
        if delta_role:
            current_delta["role"] = delta_role
        if delta_content:
            current_delta["content"] = delta_content
        
        chunk_choices = [{
            "index": 0,
            "delta": current_delta,
            "finish_reason": finish_reason
        }]
        
        response_chunk_dict = {
            "id": chunk.id,
            "object": "chat.completion.chunk",
            "created": chunk.created,
            "model": chunk.model,
            "choices": chunk_choices,
            "citations": citations_list
        }
        
        yield f"data: {json.dumps(response_chunk_dict)}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions", tags=["Completions"])
async def chat_completions_endpoint(request: Request):
    """
    OpenAI-compatible chat completions endpoint with RAG and citations.
    Supports streaming and non-streaming responses.
    """
    try:
        request_data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    messages = request_data.get("messages")
    if not messages or not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="No messages provided or messages format is invalid.")

    last_message = messages[-1]
    if not isinstance(last_message, dict) or "content" not in last_message:
        raise HTTPException(status_code=400, detail="Last message is malformed.")
    
    user_query = last_message.get("content")
    if not user_query or not isinstance(user_query, str):
        raise HTTPException(status_code=400, detail="Last message content is empty or not a string.")

    is_stream = request_data.get("stream", False)
    if not isinstance(is_stream, bool):
        is_stream = False

    logger.info(f"Received chat completion request. Stream: {is_stream}. Query: '{user_query[:50]}...'")

    try:
        citations, context = await _get_documents_and_context(user_query)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting documents and context: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve or process documents.")

    answer = citations[0]["meta"].get("answer", "") if citations else ""
    
    if settings.FAQ_ENABLE_PARAPHRASING or not citations or not answer:
        llm = LLM()

        if is_stream:
            try:
                llm_client_stream = await llm.get_answer_async_stream(
                    query=user_query, 
                    context=context
                )
                generator = _stream_response_generator(
                    llm_client_stream=llm_client_stream,
                    citations_list=citations
                )
                return StreamingResponse(generator, media_type="text/event-stream")

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                if await request.is_disconnected():
                    logger.warning("Client disconnected during streaming error handling.")
                    return 
                raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")
        else:
            try:
                llm_answer_content = await llm.get_answer_async(
                    query=user_query, 
                    context=context
                )
                response_dict = llm_answer_content.to_dict()
                response_dict["citations"] = citations
                return response_dict
            except Exception as e:
                logger.error(f"Non-streaming error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate completion: {str(e)}")
            
    else:
        if is_stream:
            json_response = {
                "id": str(uuid.uuid4()),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": settings.LLM_OPENAI_MODEL,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }],
                "citations": citations
            }
            async def async_stream_response():
                yield f"data: {json.dumps(json_response)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(async_stream_response(), media_type="text/event-stream")
        else:
            json_response = {
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": settings.LLM_OPENAI_MODEL,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }],
                "citations": citations
            }
            return json_response
