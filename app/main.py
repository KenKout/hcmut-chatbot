import uvicorn
import argparse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.envs import settings
from app.routers import query, upload, completions
from app.database import Database
from loguru import logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.APP_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router, tags=["query"])
app.include_router(upload.router, tags=["File Uploads"])
app.include_router(completions.router, tags=["Completions"])

def main():
    parser = argparse.ArgumentParser(description="Main application CLI")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex the database with specified FAQ and Web files.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run reindexing in development mode (smaller dataset).",
    )
    args = parser.parse_args()

    if args.reindex:
        logger.info("Starting reindexing process...")
        faq_file_path = input("Enter the path to the FAQ file (e.g., data/faq.csv): ")
        web_file_path = input("Enter the path to the Web data file (e.g., data/web.json): ")
        
        try:
            database = Database(recreate_index=True)
            database.reindex(faq_file_path, web_file_path, dev=args.dev)
            logger.info("Database reindexing completed successfully.")
        except FileNotFoundError as e:
            logger.error(f"Error during reindexing: {e}. Please check file paths.")
        except KeyError as e:
            logger.error(f"Error during reindexing: {e}. Please check file content and column names.")
        except ValueError as e:
            logger.error(f"Error during reindexing: {e}. Please check file formats (must be CSV or JSON).")
        except Exception as e:
            logger.error(f"An unexpected error occurred during reindexing: {e}")
    else:
        uvicorn.run(
            app,
            host=settings.APP_HOST,
            port=settings.APP_PORT,
            workers=settings.APP_WORKERS,
        )

if __name__ == "__main__":
    main()
