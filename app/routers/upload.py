import shutil
import tempfile
from pathlib import Path
from typing import List, Annotated

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from loguru import logger

from app.models.upload import FileUploadParams, FileUploadRequest
from app.utils.pipelines import FileProcessingPipeline

router = APIRouter()

def get_file_upload_params(
    split_by: Annotated[str, Form(description="Unit by which to split the documents.")] = "word",
    split_length: Annotated[int, Form(description="The maximum number of units in each split.")] = 200,
    split_overlap: Annotated[int, Form(description="The number of units that each split overlaps with the previous one.")] = 0,
    remove_empty_lines: Annotated[bool, Form(description="Whether to remove empty lines from the documents.")] = True,
    remove_extra_whitespaces: Annotated[bool, Form(description="Whether to remove extra whitespaces from the documents.")] = True,
) -> FileUploadParams:
    return FileUploadParams(
        split_by=split_by,
        split_length=split_length,
        split_overlap=split_overlap,
        remove_empty_lines=remove_empty_lines,
        remove_extra_whitespaces=remove_extra_whitespaces,
    )


@router.post("/upload-files/", summary="Upload and process multiple files")
async def upload_and_process_files(
    files: List[UploadFile] = File(..., description="List of files to upload and process."),
    params: FileUploadParams = Depends(get_file_upload_params),
):
    """
    Uploads one or more files, processes them using a Haystack pipeline,
    and ingests them into the 'files' document store.

    The processing parameters for the `DocumentPreprocessor` can be configured.
    Supported file types: .txt, .csv, .md, .pdf, .docx, .pptx.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for uploaded files: {temp_dir}")
    processed_files_paths: List[Path] = []
    results = []

    try:
        for uploaded_file in files:
            if not uploaded_file.filename:
                logger.warning("Received a file without a filename. Skipping.")
                continue

            file_path = Path(temp_dir) / uploaded_file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)
            logger.info(f"Saved uploaded file: {file_path}")
            processed_files_paths.append(file_path)
            await uploaded_file.close() # Ensure file is closed

        if not processed_files_paths:
            raise HTTPException(status_code=400, detail="No valid files were processed (e.g., all files might have been missing filenames).")

        logger.info(f"Initializing FileProcessingPipeline with params: {params.model_dump_json()}")
        file_pipeline = FileProcessingPipeline(file_upload_params=params)
        
        logger.info(f"Running pipeline for files: {processed_files_paths}")
        pipeline_result = file_pipeline.run(file_paths=processed_files_paths)
                
        results.append({
            "message": f"Successfully processed {len(processed_files_paths)} files.",
            "processed_files": [str(p.name) for p in processed_files_paths],
            "pipeline_output_keys": list(pipeline_result.keys()) if pipeline_result else []
        })
        logger.info(f"Files processed successfully: {[str(p.name) for p in processed_files_paths]}")

    except Exception as e:
        logger.error(f"Error during file processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during file processing: {str(e)}")
    finally:
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    return {"detail": "Files processed and indexed successfully.", "results": results}
