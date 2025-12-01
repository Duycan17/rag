# Document Chat RAG API

A RAG (Retrieval-Augmented Generation) API that enables users to chat with their uploaded documents using AI.

## Base URL

```
http://ragmobile.duckdns.org
```

## Interactive Documentation

Swagger UI: [http://ragmobile.duckdns.org/docs](http://ragmobile.duckdns.org/docs)

---

## Endpoints

### 1. Chat with Document

Chat with a processed document using natural language questions.

**Endpoint:** `POST /api/chat`

**Request:**
```bash
curl -X POST http://ragmobile.duckdns.org/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "c75fb994-287e-4553-b44b-67c05d7bb3eb",
    "document_id": "9926dd9c-346d-4a2a-b6b0-ffe6d5a0d942",
    "message": "What is this document about?"
  }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | UUID | Yes | UUID of the user |
| `document_id` | UUID | Yes | UUID of the document to query |
| `message` | string | Yes | The question to ask (min 1 character) |

**Response (200 OK):**
```json
{
  "answer": "This document is about...",
  "sources": [
    {
      "content": "Relevant text chunk from document...",
      "metadata": {
        "chunk_index": 0,
        "document_id": "9926dd9c-346d-4a2a-b6b0-ffe6d5a0d942"
      }
    }
  ]
}
```

---

### 2. Process Document

Process a document to create embeddings before chatting. Must be called before using the chat endpoint.

**Endpoint:** `POST /api/documents/process`

**Request:**
```bash
curl -X POST http://ragmobile.duckdns.org/api/documents/process \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "c75fb994-287e-4553-b44b-67c05d7bb3eb",
    "document_id": "9926dd9c-346d-4a2a-b6b0-ffe6d5a0d942"
  }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | UUID | Yes | UUID of the document owner |
| `document_id` | UUID | Yes | UUID of the document to process |

**Response (200 OK):**
```json
{
  "status": "ready",
  "chunks_created": 15,
  "document_id": "9926dd9c-346d-4a2a-b6b0-ffe6d5a0d942"
}
```

---

### 3. Health Check

Check if the API is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy"
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "ErrorType",
  "detail": "Detailed error message",
  "code": "ERROR_CODE"
}
```

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 403 | `AUTHORIZATION_ERROR` | User doesn't own the document |
| 404 | `NOT_FOUND` | Document not found |
| 422 | `VALIDATION_ERROR` | Invalid request parameters |
| 500 | `PROCESSING_ERROR` | Document processing failed |

---

## Usage Flow

1. **Upload a document** to Supabase `user_docs` table with a `file_url`
2. **Process the document** using `/api/documents/process`
3. **Chat with the document** using `/api/chat`

---

## Supported File Types

- PDF (.pdf)
- Plain Text (.txt)
