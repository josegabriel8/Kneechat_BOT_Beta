## KneeChat — RAG chatbot for total knee arthroplasty patients

KneeChat (ATRbot) is a research/prototype Retrieval-Augmented Generation (RAG) chatbot designed to help patients undergoing total knee arthroplasty (artroplastia total de rodilla). It answers common questions about the procedure, preparation, rehabilitation and emotional support by retrieving relevant passages from a curated document collection and generating empathetic responses.

IMPORTANT: this repository contains a preliminary evaluation version. It uses Groq (via the `langchain_groq` integration) for LLM inference in the development/evaluation pipeline. The final production version will run on a private GPU server (planned) and may use a different LLM hosting arrangement.

## Project status

- Preliminary/evaluation: intended for testing, validation and internal evaluation only.
- Not for clinical decision making. Do not use as a substitute for professional medical advice.

## Repository layout

- `src/` — application code
  - `app_groq.py` — primary entrypoint for the Groq-based bot version (uses `TELEGRAM_TOKEN2`, `GROQ_API_KEY`).
  - `bot/chatbot.py` — Telegram bot handlers and chaining logic (conversational RAG + memory).
  - `bot/rag_pipeline.py` — document ingestion, FAISS index creation and retrievers.
  - `bot/welcome.py` — welcome message constants.
- `data/` — source documents (example: `articulos 2025/`).
- `faiss_index/`, `faiss_index2/` — pre-built FAISS indexes (binary index files).
- `ingestion/` — document ingestion helpers.
- `requirements.txt` — Python dependencies.
- `.env` — local environment variable template (should NOT be committed with secrets).

## Short contract (what this repo does)

- Input: user messages from Telegram chats.
- Output: short empathetic Spanish responses with cited sources (when available).
- Error modes: gracefully falls back with maintenance message when LLM/Groq API fails.
- Success: returns relevant, concise responses and a small set of source citations.

## Quick start (development / evaluation)

1. Clone the repository.
2. Create a virtual environment and install dependencies.

PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

3. Environment variables

Create a `.env` file in the repository root (the project already includes a `.env` template). At minimum, set:

- `GROQ_API_KEY` — Groq API key used by `langchain_groq`.
- `TELEGRAM_TOKEN2` — Telegram bot token used by the Groq bot entrypoint (`app_groq.py`).
- `HF_TOKEN` — (optional) Hugging Face token for embeddings/models if needed.
- `OPENAI_API_KEY` — (optional) present in template if other LLMs are used.

PowerShell example to set temporarily for a session:

```powershell
$env:GROQ_API_KEY = "your_groq_key"
$env:TELEGRAM_TOKEN2 = "your_telegram_token"
$env:HF_TOKEN = "your_hf_token"
```

4. Run the bot (Groq-based entrypoint)

```powershell
python src\app_groq.py
```

Notes:
- `app_groq.py` explicitly checks for `GROQ_API_KEY` and `TELEGRAM_TOKEN2` and will raise an error if they are not present.
- The bot uses `langchain_groq.ChatGroq` configured with `llama-3.1-8b-instant` in the prototype.

## Indexing and data ingestion

- The ingestion script splits documents (PDF / JSON), deduplicates chunks and builds FAISS indexes using `sentence-transformers/LaBSE` embeddings (CPU by default).
- Default data path expected: `data/` (see `src/bot/rag_pipeline.py`). A priority document named `Main Knowledge.pdf` is treated as the primary source.
- To (re)build indexes:

```powershell
python src\bot\rag_pipeline.py
```

This will create local FAISS folders such as `faiss_index`, `faiss_index_priority` and `faiss_index_secondary`.

## How RAG works here (brief)

- Ingestion: PDFs/JSONs are parsed, split into large chunks (~1500 chars), and deduplicated.
- Embeddings: LaBSE embeddings via HuggingFace are used to index chunks into FAISS.
- Retrieval: The retriever searches priority sources first (hierarchical retriever). The conversational chain combines retrieved context with a Groq LLM to generate answers.

## Evaluation & metrics

- The bot collects a simple abstention metric (counts instances where the bot responds "No tengo información específica...").
- This repository is structured for evaluation; expect to run manual experiments to measure accuracy, recall of sources, and safety.

## Privacy and safety

- This project processes patient-facing materials. Ensure you have the right to use and publish source documents.
- Do NOT deploy this evaluation system in production with real patient data until privacy, security and clinical validation are completed.

## Notes about current implementation

- The current prototype uses Groq via `langchain_groq` for inference (this is the preliminary evaluation setup).
- The final plan is to run the model on a private GPU server (higher throughput and lower latency). Expect changes to how API keys, model endpoints, and costs are handled in the production deployment.

## Troubleshooting

- If you get authentication errors, double-check `GROQ_API_KEY` and `TELEGRAM_TOKEN2` values.
- If FAISS index creation is slow or out-of-memory, consider running on a machine with more RAM or using smaller chunk sizes and batching.