# FAQ Workflow Diagram

```mermaid
flowchart LR
  %% =========================
  %% Ingestion Pipeline
  %% =========================
  subgraph Ingestion [Ingestion Pipeline]
    A[PDF Upload or URL\nPOST /ingest-pdf] --> B[Save PDF\n/data/uploads]
    B --> C[Extract Text\nfitz (PyMuPDF)]
    C --> D[Generate Structured FAQ\nLLM (ChatOpenAI)]
    D --> E[Enrich + Link Map\nIDs, link_id/link_url]
    E --> F[Store Structured FAQ\nMongoDB: structured_faqs]
    E --> G[Store Links\nMongoDB: link_metadata]
    E --> H[Embed Questions\nChroma: questions]
    E --> I[Embed RAG Docs\nChroma: rag]
  end

  %% =========================
  %% Runtime API
  %% =========================
  subgraph Runtime [Runtime API]
    U[Frontend / Client] -->|POST /chat| J[InsuranceQueryEngine]
    U -->|GET /predefined-questions| K[MongoDB: structured_faqs]
    U -->|POST /suggest| L[SuggestionEngine]
    U -->|GET /session/{id}/history| M[SessionMemory]
    U -->|DELETE /session/{id}| M

    J --> N[StructuredFAQEngine\nVector search (questions)]
    N -->|Match >= threshold| O[MongoDB: structured_faqs]
    O -->|Answer Blocks + Related| P[Structured FAQ Response]

    J -->|Fallback| Q[RAGEngine\nVector search (rag)]
    Q --> R[LLM (ChatOpenAI)]
    Q --> S[MongoDB: link_metadata]
    R --> T[RAG Response]

    L --> X[Chroma: questions + rag]
    X --> Y[Keyword + Semantic Merge]
    Y --> Z[Suggestions Response]

    J --> M[SessionMemory\n(topic, last question, history)]
  end

  %% =========================
  %% Shared Data Stores
  %% =========================
  Mongo[(MongoDB)]
  Chroma[(ChromaDB)]

  O --- Mongo
  K --- Mongo
  S --- Mongo
  F --- Mongo
  G --- Mongo

  H --- Chroma
  I --- Chroma
  N --- Chroma
  Q --- Chroma
  X --- Chroma
```

Notes:
- Structured FAQ path is preferred; RAG is used only if no strong FAQ match is found.
- Suggestions use both question and RAG vector stores, then merge keyword + semantic hits.
