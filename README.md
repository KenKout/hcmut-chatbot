# HCMUT Chatbot Application

This application provides a backend API for a chatbot, featuring document indexing, querying capabilities, and file uploads. It uses FastAPI for the web framework and Qdrant as the vector database for semantic search.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create and Activate a Virtual Environment](#2-create-and-activate-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Configure Environment Variables](#4-configure-environment-variables)
  - [5. Set Up Qdrant](#5-set-up-qdrant)
  - [6. Set Up Hugging Face Text Embeddings Inference (TEI) Server](#6-set-up-hugging-face-text-embeddings-inference-tei-server)
  - [7. Set Up Hugging Face Text Generation Inference (TGI) Server](#7-set-up-hugging-face-text-generation-inference-tgi-server)
  - [8. Deploying with Docker Compose](#8-deploying-with-docker-compose)
- [Running the Application (Locally, without Docker Compose for the app)](#running-the-application-locally-without-docker-compose-for-the-app)
- [Reindexing Data](#reindexing-data)
- [Data Formats](#data-formats)
  - [FAQ Data (`faq.csv` or `faq.json`)](#faq-data-faqcsv-or-faqjson)
  - [Web Data (`web.json` or `web.csv`)](#web-data-webjson-or-webcsv)
- [API Endpoints](#api-endpoints)

## Features

-   **FastAPI Backend**: Modern, fast (high-performance) web framework for building APIs.
-   **Qdrant Vector Database**: Efficient similarity search for document retrieval.
-   **Hugging Face Embeddings**: Supports various embedding providers, including self-hosted TEI.
-   **Data Reindexing**: CLI command to reindex FAQ and web data.
-   **File Uploads**: Endpoint for uploading and processing files (details to be implemented).

## Project Structure

```
.
├── app/                      # Main application module
│   ├── __init__.py
│   ├── database.py           # Qdrant database interaction, reindexing logic
│   ├── envs.py               # Environment variable settings
│   ├── main.py               # FastAPI application entry point, CLI for reindex
│   ├── models/               # Pydantic models for API requests/responses
│   │   ├── embedding.py
│   │   ├── query.py
│   │   └── upload.py
│   ├── routers/              # API routers
│   │   ├── query.py
│   │   └── upload.py
│   └── utils/                # Utility modules
│       ├── embedders.py      # Embedding component setup
│       ├── llm.py            # LLM interaction (e.g., for paraphrasing)
│       └── pipelines.py      # Haystack query pipelines
├── data/                     # Sample data files (you might need to create this)
│   ├── .gitignore
│   └── (example: hcmut_data_faq.csv)
│   └── (example: hcmut_data_web.json)
├── .env.example              # Example environment variables
├── .gitignore
├── Dockerfile                # Dockerfile for the main application
├── docker-compose.yml        # Docker Compose configuration
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Prerequisites

-   Python 3.8+
-   Pip (Python package installer)
-   Docker and Docker Compose (for Qdrant, TEI server, and application deployment)
-   Access to an OpenAI API key (if using OpenAI models) or Hugging Face API key (if using HF Inference API for non-self-hosted TEI)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/KenKout/hcmut-chatbot
cd hcmut-chatbot
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and update it with your specific configurations:

```bash
cp .env.example .env
```

Refer to the comments in [`/.env.example`](./.env.example:1) for detailed explanations of each variable.

### 5. Set Up Qdrant

If `DEBUG=true`, Qdrant runs in-memory, and no external setup is needed.
For production (`DEBUG=false`), you need a running Qdrant instance. You can run Qdrant using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
This command mounts a local directory `qdrant_storage` to persist Qdrant data.
Ensure your `QDRANTDB_URL` in the `.env` file points to this instance (e.g., `http://localhost:6333`).

### 6. Set Up Hugging Face Text Embeddings Inference (TEI) Server

If you choose `EMBEDDING_PROVIDER="huggingface"` and `EMBEDDING_HUGGINGFACE_API_TYPE="text_embeddings_inference"`, you need to run a TEI server. This allows you to self-host open-source embedding models.

**Example using Docker for `bkai-foundation-models/vietnamese-bi-encoder` (CPU):**

```bash
docker run -p 8080:80 \
    --pull always \
    ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
    --model-id bkai-foundation-models/vietnamese-bi-encoder
```

**For GPU support (requires NVIDIA drivers and NVIDIA Container Toolkit):**

```bash
docker run -p 8080:80 --gpus all \
    --pull always \
    ghcr.io/huggingface/text-embeddings-inference:latest \
    --model-id bkai-foundation-models/vietnamese-bi-encoder
```

-   The TEI server will be available at `http://localhost:8080`.
-   Update `EMBEDDING_HUGGINGFACE_BASE_URL="http://localhost:8080"` in your `.env` file.
-   The `SentenceTransformersDocumentEmbedder` in Haystack can then use this TEI server by configuring its `api_type` to `text_embeddings_inference` and providing the `api_key` (if your TEI server is secured, though the default Docker command does not set an API key) and `url`. Our application's [`app/utils/embedders.py`](app/utils/embedders.py:1) handles this configuration based on environment variables.

You can replace `bkai-foundation-models/vietnamese-bi-encoder` with any other Sentence Transformer model compatible with TEI. Check the [Text Embeddings Inference documentation](https://huggingface.co/docs/text-embeddings-inference/index) for more models and advanced configurations.

### 7. Set Up Hugging Face Text Generation Inference (TGI) Server

If you want to self-host a Large Language Model (LLM) for tasks like paraphrasing or generation, you can use Hugging Face's Text Generation Inference (TGI). The application can be configured to use a TGI endpoint as an OpenAI-compatible API.

**Example using Docker (CPU):**

```bash
# Replace your-model-id with the desired Hugging Face model, e.g., gpt2 or a larger model
# Ensure the model is compatible with TGI.
docker run -p 8081:80 --pull always \
    -v $(pwd)/tgi_cache:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id your-model-id --port 80
```

**For GPU support (requires NVIDIA drivers and NVIDIA Container Toolkit):**

```bash
docker run -p 8081:80 --gpus all --pull always \
    -v $(pwd)/tgi_cache:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id your-model-id --port 80
    # Add other TGI flags as needed, e.g., --num-shard, --quantize
```

-   The TGI server will be available at `http://localhost:8081` (or the host port you map).
-   Update `LLM_OPENAI_BASE_URL="http://localhost:8081/v1"` in your `.env` file if you are running TGI locally and want the app to use it. The `/v1` path is often used for OpenAI compatibility.
-   Set `LLM_MODEL_ID` in your `.env` to the model ID you are serving with TGI (this is for reference, the actual model served is determined by the TGI command).
-   TGI can serve models in an OpenAI-compatible way. Refer to the [TGI documentation](https://huggingface.co/docs/text-generation-inference/index) for details on compatible models and advanced configurations (like quantization, sharding for large models, etc.).

### 8. Deploying with Docker Compose

The easiest way to run the entire stack (application, Qdrant, TEI server, and TGI server) is using Docker Compose.

1.  **Ensure Docker and Docker Compose are installed.**
2.  **Configure Environment Variables:** Make sure your `.env` file is correctly set up as described in [Step 4](#4-configure-environment-variables). The `docker-compose.yml` file will use these variables.
    - `QDRANTDB_URL` should be `http://qdrant_db:6333` (service name from `docker-compose.yml`).
    - `EMBEDDING_HUGGINGFACE_BASE_URL` should be `http://tei_server:80` (service name from `docker-compose.yml`).
    - `LLM_OPENAI_BASE_URL` should be `http://tgi_server:80/v1` (or your TGI service name and port, with `/v1` for OpenAI compatibility if TGI is configured for it) if you are using the self-hosted TGI server.
    - `EMBEDDING_MODEL` in `.env` will be used by the `tei_server` in `docker-compose.yml`.
    - `LLM_MODEL_ID` in `.env` will be used by the `tgi_server` in `docker-compose.yml` (ensure this model is compatible and suitable for your TGI setup).
    - `TGI_HTTP_PORT` in `.env` can be used to configure the host port for the TGI service (e.g., `TGI_HTTP_PORT=8081`).
    - Update your `.env` file accordingly.

3.  **Build and Run the Services:**
    ```bash
    docker-compose up --build
    ```
    To run in detached mode (in the background):
    ```bash
    docker-compose up --build -d
    ```

4.  **Accessing the Application:**
    -   The API will be available at `http://<APP_HOST>:<APP_PORT>` (e.g., `http://localhost:8000` if `APP_HOST=0.0.0.0` and `APP_PORT=8000` in your `.env`).
    -   Qdrant will be accessible on the host at `http://localhost:6333`.
    -   The TEI server will be accessible on the host at `http://localhost:8080`.
    -   The TGI server will be accessible on the host at `http://localhost:${TGI_HTTP_PORT:-8081}` (e.g., `http://localhost:8081`).

5.  **Stopping the Services:**
    ```bash
    docker-compose down
    ```
    To remove volumes (Qdrant data, TEI cache) as well:
    ```bash
    docker-compose down -v
    ```

**Note on Reindexing with Docker Compose:**
If you need to reindex data while using Docker Compose, you can execute the reindex command inside the running `app` container:
```bash
docker-compose exec app python -m app.main --reindex
```
You will be prompted to enter file paths. These paths should be accessible from *within the container's filesystem*. If your data files are in the `data/` directory and this directory is mounted as a volume (as configured in the provided `docker-compose.yml`), you can use paths like `data/hcmut_data_faq.csv`.

## Running the Application (Locally, without Docker Compose for the app)

Once the setup is complete, you can run the FastAPI application:

```bash
python -m app.main
```

The API will be available at `http://<APP_HOST>:<APP_PORT>` (e.g., `http://localhost:8000` by default).
You can access the OpenAPI documentation at `http://localhost:8000/docs`.

## Reindexing Data

The application provides a CLI command to reindex data into Qdrant. This is useful when you have new or updated FAQ or web data files.

To run the reindexing process:

```bash
python -m app.main --reindex
```

The script will prompt you to enter the paths for your FAQ data file and web data file.

**Example:**
```
python -m app.main --reindex
# Output:
# Starting reindexing process...
# Enter the path to the FAQ file (e.g., data/faq.csv): data/hcmut_data_faq.csv
# Enter the path to the Web data file (e.g., data/web.json): data/hcmut_data_web.json
# ... (processing logs) ...
# Database reindexing completed successfully.
```

You can also use the `--dev` flag to reindex a smaller subset of the data, which is useful for development and testing:

```bash
python -m app.main --reindex --dev
```

Ensure your data files are in the correct format (see [Data Formats](#data-formats)).

## Data Formats

The reindexing process expects specific formats for FAQ and web data.

### FAQ Data (`faq.csv` or `faq.json`)

-   **CSV Format:** Must contain `query` and `answer` columns.
    ```csv
    query,answer
    "What are the admission requirements?","The admission requirements include a high school diploma and standardized test scores."
    "What is the tuition fee?","The tuition fee varies by program. Please check the university website."
    ```

-   **JSON Format:** An array of objects, each with `query` and `answer` keys.
    ```json
    [
      {
        "query": "What are the admission requirements?",
        "answer": "The admission requirements include a high school diploma and standardized test scores."
      },
      {
        "query": "What is the tuition fee?",
        "answer": "The tuition fee varies by program. Please check the university website."
      }
    ]
    ```

### Web Data (`web.json` or `web.csv`)

This format is typically used for scraped website content.

-   **JSON Format:** An array of objects, each with `text` (main content) and `tables` (array of strings, each string being a table serialized as text, e.g., Markdown or HTML).
    ```json
    [
      {
        "text": "This is the main content of page 1. It discusses various academic programs.",
        "tables": [
          "| Program | Duration | Credits |\n|---|---|---|\n| CS | 4 years | 120 |",
          "| EE | 4 years | 124 |"
        ]
      },
      {
        "text": "Page 2 talks about campus life and facilities.",
        "tables": []
      }
    ]
    ```

-   **CSV Format:** Must contain `text` and `tables` columns. The `tables` column should be a string representation of a list of table strings (e.g., a JSON-encoded string of a list).
    ```csv
    text,tables
    "This is the main content of page 1. It discusses various academic programs.","[\"| Program | Duration | Credits |\\n|---|---|---|\\n| CS | 4 years | 120 |\", \"| EE | 4 years | 124 |\"]"
    "Page 2 talks about campus life and facilities.","[]"
    ```
    *Note: Storing complex structures like lists of tables in CSV can be cumbersome. JSON is generally preferred for web data.*

## API Endpoints

The application exposes the following main API endpoints (details can be found in the OpenAPI docs at `/docs` when the app is running):

-   **`/query/` (POST)**: Submit a query to get relevant information from the indexed documents.
-   **`/upload-file/` (POST)**: Upload a file for processing (specific processing logic depends on implementation in [`app/routers/upload.py`](app/routers/upload.py:1)).

Refer to [`app/routers/query.py`](app/routers/query.py:1) and [`app/routers/upload.py`](app/routers/upload.py:1) for more details on request/response models.