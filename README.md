# RAG_Playground

A proof-of-concept Retrieval-Augmented Generation (RAG) playground built with Flask, LangChain, and OpenAI/Tavily. This project demonstrates how to combine PDF retrieval, web search, and conversational agents into a simple chat interface.

## Features

- `AgenticRAG`: loads PDFs, applies adaptive chunking, builds a FAISS vector store, and uses a LangChain agent with tool support.
- PDF search tool for retrieving relevant document chunks.
- Web search tool powered by Tavily for up-to-date answers when PDFs do not contain enough information.
- Conversation memory for chat history and context.
- Simple Flask frontend with a chat UI at `/`.
- Additional hybrid retrieval utilities for BM25 and metrics evaluation.

## Repository Structure

- `app.py` - Flask web app exposing chat, clear memory, and history endpoints.
- `agenticrag.py` - Main RAG implementation using LangChain agents and tools.
- `adaptivechunking.py` - Adaptive text chunking utility for document splitting.
- `hybridrag.py` - Hybrid retrieval support combining FAISS and BM25.
- `ragmetrics.py` - Evaluation metrics for RAG retrieval performance.
- `templates/chat.html` - Web UI for chat interaction.
- `requirements.txt` - Python dependencies.

## Requirements

- Python 3.10+ recommended
- `pip` for package installation
- OpenAI API key
- Tavily API key

## Setup

1. Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your API keys inside `app.py` or via environment variables.

4. Place your PDF documents in the repository root or update the `pdf_files` list in `app.py`.

## Running the App

Start the Flask server:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser to use the chat UI.

## Usage

- Ask questions in the chat box.
- The agent will search uploaded PDFs first and fallback to web search if needed.
- Use the `/clear` endpoint to reset conversation memory.
- The `/history` endpoint returns chat history as JSON.

## Example

Update the PDF files and keys in `app.py`:

```python
rag = AgenticRAG(
    pdf_files=["./document1.pdf", "./document2.pdf"],
    openai_api_key="YOUR_OPENAI_API_KEY",
    tavily_api_key="YOUR_TAVILY_API_KEY",
    verbose=False,
)
```

Then run the app and ask questions about the uploaded documents.

## Notes

- The app currently expects PDF files in the root folder.
- This project is designed for experimentation and demonstration, not production deployment.
- `hybridrag.py` and `ragmetrics.py` provide additional retrieval and evaluation tools for advanced RAG experimentation.

## License

This repository does not include a license file. Add one if you intend to open source or share the project.
