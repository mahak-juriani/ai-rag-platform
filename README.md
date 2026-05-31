# AI RAG Platform

A Retrieval-Augmented Generation (RAG) platform built with FastAPI, ChromaDB, Groq LLMs, and semantic search. The system allows users to upload PDF documents and ask natural language questions based on the document content.

## Features

* PDF document upload and processing
* Automatic text extraction from PDFs
* Intelligent text chunking
* Vector-based document retrieval using ChromaDB
* Context-aware question answering using Groq LLMs
* REST APIs built with FastAPI
* Dockerized deployment
* Deployable on cloud platforms such as Render

---

## Tech Stack

### Backend

* FastAPI
* Python

### AI & Retrieval

* Groq API
* ChromaDB
* Retrieval-Augmented Generation (RAG)

### Deployment

* Docker
* Render

### Document Processing

* PyPDF

---

## System Architecture

PDF Upload
→ Text Extraction
→ Text Chunking
→ ChromaDB Storage
→ Semantic Retrieval
→ Groq LLM
→ Generated Response

---

## API Endpoints

### Health Check

GET /

Returns application status.

### Upload Document

POST /upload

Uploads and processes PDF documents.

### Query Documents

GET /chat?query=<question>

Returns an AI-generated response using relevant document context.

### Semantic Search

GET /search?query=<question>

Retrieves the most relevant document chunks from the vector database.

---

## Project Workflow

1. User uploads a PDF document.
2. Text is extracted from the PDF.
3. Extracted text is split into smaller chunks.
4. Chunks are stored in ChromaDB.
5. User submits a question.
6. Relevant chunks are retrieved using semantic search.
7. Retrieved context is sent to a Groq-hosted LLM.
8. The generated answer is returned to the user.

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/mahak-juriani/ai-rag-platform.git
cd ai-rag-platform
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Run the application:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Running With Docker

Build image:

```bash
docker build -t ai-rag-platform .
```

Run container:

```bash
docker run -p 8000:8000 --env-file .env ai-rag-platform
```

---

## Future Improvements

* Multi-document knowledge base management
* Authentication and user accounts
* Conversation history
* Streaming LLM responses
* Source citation support
* Cloud-based vector storage
* Advanced metadata filtering

---

## Learning Outcomes

This project helped explore:

* FastAPI backend development
* Retrieval-Augmented Generation (RAG)
* Vector databases
* LLM integration
* Semantic search
* Docker-based deployment
* API design and development

---

## License

This project is intended for learning, experimentation, and demonstrating backend AI engineering concepts.
