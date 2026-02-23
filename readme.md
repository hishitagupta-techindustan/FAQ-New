# FAQ Workflow Diagram

```mermaid
flowchart LR
  %% =========================
  %% Ingestion Pipeline
  %% =========================
  subgraph Ingestion [Ingestion Pipeline]
    A["PDF Upload or URL<br/>POST /ingest-pdf"] --> B["Save PDF<br/>/data/uploads"]
    B --> C["Extract Text<br/>fitz PyMuPDF"]
    C --> D["Generate Structured FAQ<br/>LLM ChatOpenAI"]
    D --> E["Enrich + Link Map<br/>IDs, link_id/link_url"]
    E --> F["Store Structured FAQ<br/>MongoDB: structured_faqs"]
    E --> G["Store Links<br/>MongoDB: link_metadata"]
    E --> H["Embed Questions<br/>Chroma: questions"]
    E --> I["Embed RAG Docs<br/>Chroma: rag"]
  end

  %% =========================
  %% Runtime API
  %% =========================
  subgraph Runtime [Runtime API]
    U["Frontend / Client"] -->|POST /chat| J["InsuranceQueryEngine"]
    U -->|GET /predefined-questions| K["MongoDB: structured_faqs"]
    U -->|POST /suggest| L["SuggestionEngine"]
    U -->|GET /session/:id/history| M["SessionMemory"]
    U -->|DELETE /session/:id| M

    J --> N["StructuredFAQEngine<br/>Vector search questions"]
    N -->|Match >= threshold| O["MongoDB: structured_faqs"]
    O -->|Answer Blocks + Related| P["Structured FAQ Response"]

    J -->|Fallback| Q["RAGEngine<br/>Vector search rag"]
    Q --> R["LLM ChatOpenAI"]
    Q --> S["MongoDB: link_metadata"]
    R --> T["RAG Response"]

    L --> X["Chroma: questions + rag"]
    X --> Y["Keyword + Semantic Merge"]
    Y --> Z["Suggestions Response"]

    J --> M["SessionMemory<br/>topic, last question, history"]
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

## Chroma Cloud Setup

To use Chroma Cloud (recommended for deployment), set these environment variables:

```
CHROMA_API_KEY=your_key
CHROMA_TENANT=your_tenant
CHROMA_DATABASE=your_database
```

If these are not set, the app falls back to local Chroma storage at the configured `chroma_persist_directory`.

After switching to Cloud, re-run ingestion so vectors are written to the cloud database.



# Suggestions Pipeline - before optimisation


                 ┌─────────────────────────┐
                 │  User types query       │
                 │  "appliance repair..."  │
                 └─────────────┬───────────┘
                               │
                               ▼
                 ┌─────────────────────────┐
                 │ 1️⃣ Semantic Retrieval   │
                 │ Chroma similarity_search │
                 │ (question + rag store)   │
                 └─────────────┬───────────┘
                               │
                               ▼
                 ┌─────────────────────────┐
                 │ Candidate Pool          │
                 │ {question, topic, score}│
                 └─────────────┬───────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
┌─────────────────────┐                    ┌─────────────────────┐
│ 2️⃣ Keyword Scoring  │                    │ 3️⃣ Semantic Filter │
│ over candidates      │                    │ keep score ≥ 0.65   │
│ substring / coverage │                    │                     │
└───────────┬─────────┘                    └───────────┬─────────┘
            ▼                                            ▼
     Keyword Hits                                  Semantic Hits
            │                                            │
            └──────────────┬─────────────────────────────┘
                           ▼
                 ┌─────────────────────────┐
                 │ 4️⃣ Merge + Deduplicate │
                 │ if same question:       │
                 │  score = 0.8S + 0.2K    │
                 │  match_type = "both"    │
                 └─────────────┬───────────┘
                               │
                               ▼
                 ┌─────────────────────────┐
                 │ 5️⃣ Sort by score DESC  │
                 └─────────────┬───────────┘
                               │
                               ▼
                 ┌─────────────────────────┐
                 │ 6️⃣ Threshold Filter     │
                 │ keep score ≥ 0.60       │
                 └─────────────┬───────────┘
                               │
                               ▼
                 ┌─────────────────────────┐
                 │ 7️⃣ Fill gaps            │
                 │ add keyword hits        │
                 │ add LLM fallback        │
                 └─────────────┬───────────┘
                               │
                               ▼
                 ┌─────────────────────────┐
                 │ Return ≤ 5 suggestions  │
                 └─────────────────────────┘


# Suggestions Pipeline : after optimisation

User
 │
 ▼
Normalize
 │
 ▼
Cache?
 ├── YES → Return
 └── NO
        │
        ▼
Inflight?
 ├── YES → Wait Future → Return
 └── NO
        │
        ▼
ThreadPool.submit(_compute)
        │
        ▼
Embed Query
        │
        ▼
Vector Search ─────┐
                    ├── Merge → Score → Top K
Keyword Search ─────┘
        │
        ▼
Optional LLM Rerank
        │
        ▼
Store in Cache
Remove from Inflight
        │
        ▼
Return Suggestions