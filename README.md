# PDF Question-Answering App

A RAG (Retrieval-Augmented Generation) app that answers questions about PDF documents.

## Status
- [x] PDF text extraction
- [x] Chunking with overlap
- [x] Embeddings + semantic search
- [ ] Vector database (ChromaDB)
- [ ] LLM answer generation
- [ ] Web UI

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python search_demo.py
```