from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.utils.pipelines import ChatPipeline
from app.envs import settings
from app.models.query import Query, QueryResponse
from app.utils.llm import LLM
from loguru import logger

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(query_request: Query):
    """
    Endpoint to handle query requests.
    """
    try:
        # Initialize the pipeline
        pipeline = ChatPipeline()
        
        # Run the pipeline with the provided query
        pipeline_output = await run_in_threadpool(pipeline.run, query_request.query)
        
        # Extract relevant information from the pipeline output
        faq_documents = pipeline_output.get("faq_documents", [])
        web_documents = pipeline_output.get("web_documents", [])
        file_documents = pipeline_output.get("file_documents", [])

        # logger.debug(f"FAQ documents: {faq_documents}")
        # logger.debug(f"Web documents: {web_documents}")
        
        if faq_documents:
            if not settings.FAQ_ENABLE_PARAPHRASING:
                return QueryResponse(
                    faq_documents=faq_documents,
                    web_documents=web_documents,
                    file_documents=file_documents,
                    llm_answer=faq_documents[0].meta.get('answer', "")
                )

        # Initialize LLM
        llm = LLM()
        
        # Get answer from LLM using the retrieved documents
        context_parts = []
        if faq_documents:
            context_parts.extend([doc.content for doc in faq_documents if doc.content])
        if web_documents:
            context_parts.extend([doc.content for doc in web_documents if doc.content])
        if file_documents:
            context_parts.extend([doc.content for doc in file_documents if doc.content])
        
        context = "\n\n".join(context_parts)
        
        llm_answer = await run_in_threadpool(llm.get_answer, query_request.query, context if context else "")
        
        return QueryResponse(
            faq_documents=faq_documents,
            web_documents=web_documents,
            file_documents=file_documents,
            llm_answer=llm_answer
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
