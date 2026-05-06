# NaijaSense AI

NaijaSense AI is a production-oriented, modular multi-agent backend that:

- simulates user reviews (Task A),
- recommends items with context and memory (Task B),
- exposes everything through FastAPI.

## Tech Stack

- Python 3.10+
- FastAPI
- Multi-agent orchestration (custom modular design, LangChain-ready)
- In-memory vector memory layer (replaceable with FAISS/Chroma)
- Docker

## Project Structure

```text
.
├── agents/                 # User modeling, review generation, recommendation agents
├── api/                    # FastAPI app and routes
├── core/                   # Main orchestrator logic
├── memory/                 # Vector store and user memory manager
├── models/                 # LLM wrappers/abstractions
├── tests/                  # API tests
├── utils/                  # Config, logger, shared schemas
├── Dockerfile
├── main.py
├── requirements.txt
└── README.md
```

## Features

- **User Modeling Agent**: infers persona style, tone, sentiment bias, and interests.
- **Review Generation Agent**: creates culturally aligned review text and rating.
- **Recommendation Agent**: ranks candidate items based on profile + retrieved memory.
- **Memory System**: stores/retrieves user interactions with vector similarity.
- **Orchestrator**: controls workflow and exposes reasoning steps for transparency.

## API Endpoints

- `GET /api/v1/health`
- `POST /api/v1/simulate-review`
- `POST /api/v1/recommend`

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn main:app --reload
```

4. Open docs:

- [http://localhost:8000/docs](http://localhost:8000/docs)

## Running Tests

```bash
pytest -q
```

## Docker Usage

Build image:

```bash
docker build -t naijasense-ai .
```

Run container:

```bash
docker run --rm -p 8000:8000 naijasense-ai
```

## Scalability Notes

- Replace `InMemoryVectorStore` with FAISS/Chroma adapter in `memory/vector_store.py`.
- Replace deterministic `LLMWrapper` with OpenAI/local model client in `models/llm_wrapper.py`.
- Add async task queue (Celery/RQ) for heavy inference workloads.
- Add persistent DB for production-grade user profile storage.

