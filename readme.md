# System Architecture

## Overview

The Insurance FAQ Chatbot uses a sophisticated RAG (Retrieval-Augmented Generation) architecture inspired by Perplexity AI, built with LangGraph for orchestration.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                      (Streamlit Web App)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Query
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SECURITY LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Input       │  │    Rate      │  │   Session    │          │
│  │Sanitization  │  │   Limiter    │  │  Validation  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Validated Query
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CACHE LAYER                                │
│              (Check for cached responses)                        │
└────────────┬────────────────────────────────┬───────────────────┘
             │ Cache Miss                     │ Cache Hit
             ▼                                ▼
┌────────────────────────────────┐    ┌──────────────────┐
│   LANGGRAPH ORCHESTRATOR       │    │  Return Cached   │
│                                │    │    Response      │
│  ┌──────────────────────────┐ │    └──────────────────┘
│  │ 1. Query Enhancement     │ │
│  │    - Add context         │ │
│  │    - Expand query        │ │
│  └──────────┬───────────────┘ │
│             │                  │
│  ┌──────────▼───────────────┐ │
│  │ 2. Query Routing         │ │
│  │    - Classify complexity │ │
│  │    - Determine strategy  │ │
│  └──────────┬───────────────┘ │
│             │                  │
│  ┌──────────▼───────────────┐ │
│  │ 3. Document Retrieval    │ │
│  │    - Semantic search     │ │
│  │    - Hybrid search       │ │
│  └──────────┬───────────────┘ │
│             │                  │
│  ┌──────────▼───────────────┐ │
│  │ 4. Reranking             │ │
│  │    - Cross-encoder       │ │
│  │    - Relevance scoring   │ │
│  └──────────┬───────────────┘ │
│             │                  │
│  ┌──────────▼───────────────┐ │
│  │ 5. Answer Generation     │ │
│  │    - LLM synthesis       │ │
│  │    - Source citation     │ │
│  └──────────┬───────────────┘ │
│             │                  │
│  ┌──────────▼───────────────┐ │
│  │ 6. Follow-up Questions   │ │
│  │    - Generate suggestions│ │
│  └──────────────────────────┘ │
└────────────┬───────────────────┘
             │
             │ Final Response
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STORAGE & MONITORING                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   ChromaDB   │  │  LangSmith   │  │    Cache     │          │
│  │  Vector DB   │  │  Monitoring  │  │   Storage    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. User Interface Layer

**Technology:** Streamlit

**Responsibilities:**
- Render chat interface
- Display sources and citations
- Show follow-up questions
- Session management
- Display metrics

### 2. Security Layer

**Components:**

a) **Input Sanitization**
- Remove HTML/scripts
- Normalize whitespace
- Truncate long inputs

b) **Prompt Injection Protection**
- Pattern matching for injection attempts
- Keyword analysis
- Special character detection

c) **Rate Limiter**
- Per-user request limits
- Time window management
- Graceful degradation

### 3. Cache Layer

**Strategy:** Two-tier caching

a) **Query Cache**
- Enhanced queries
- TTL: 2 hours

b) **Response Cache**
- Complete responses
- Session-aware
- TTL: 1 hour

c) **Retrieval Cache**
- Retrieved documents
- TTL: 1 hour

### 4. LangGraph Orchestrator

**Technology:** LangGraph state machine

**Nodes:**

1. **Query Enhancement**
   - Uses conversation history
   - Expands short queries
   - Adds context

2. **Query Routing**
   - Classifies: simple/complex/conversational
   - Determines retrieval parameters
   - Adaptive strategy selection

3. **Document Retrieval**
   - Vector similarity search
   - Hybrid search (semantic + keyword)
   - Metadata filtering

4. **Reranking**
   - Cross-encoder scoring
   - Position-aware ranking
   - Relevance optimization

5. **Answer Generation**
   - Context-aware prompting
   - Source attribution
   - Citation formatting

6. **Follow-up Generation**
   - Question suggestions
   - Topic expansion
   - User guidance

**State Management:**
```python
class AgentState(TypedDict):
    query: str
    enhanced_query: Optional[str]
    query_type: Optional[str]
    retrieved_docs: List[Dict]
    reranked_docs: List[Dict]
    answer: Optional[str]
    sources: List[Dict]
    followup_questions: List[str]
    conversation_history: List[Dict]
    error: Optional[str]
```

### 5. Retrieval System

**Vector Store:** ChromaDB

**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2

**Features:**
- Semantic similarity search
- Hybrid search (semantic + keyword)
- Metadata filtering
- Persistent storage

**Reranking:**
- Cross-encoder model
- Relevance scoring
- Top-K selection

### 6. LLM Layer

**Provider:** Anthropic Claude

**Models:**
- Primary: claude-sonnet-4-20250514
- Fast queries: Sonnet
- Complex reasoning: Can upgrade to Opus

**Parameters:**
- Temperature: 0.0 (deterministic)
- Max tokens: 4000
- Streaming: Supported

### 7. Monitoring & Observability

**LangSmith Integration:**
- Trace all LLM calls
- Track latency
- Monitor token usage
- Error tracking

**Metrics:**
- Cache hit rate
- Query types distribution
- Average response time
- Error rates

## Data Flow

### Ingestion Flow

```
PDF Files
    │
    ▼
PDF Processor
    │ (Extract text)
    ▼
Text Chunks
    │ (Chunk with overlap)
    ▼
Embeddings
    │ (Generate vectors)
    ▼
ChromaDB
    │ (Persist)
    ▼
Vector Store
```

### Query Flow

```
User Query
    │
    ▼
Security Check → [REJECT if malicious]
    │
    ▼
Cache Check → [RETURN if hit]
    │
    ▼
Query Enhancement
    │
    ▼
Routing (Classify)
    │
    ▼
Retrieval (10-15 docs)
    │
    ▼
Reranking (Top 5)
    │
    ▼
LLM Generation
    │
    ▼
Follow-ups
    │
    ▼
Cache + Return
```

## Performance Optimizations

1. **Caching**
   - Three-tier cache strategy
   - Reduces redundant processing
   - Session-aware caching

2. **Batch Processing**
   - Batch embedding generation
   - Parallel document processing

3. **Smart Routing**
   - Simple queries → fewer docs
   - Complex queries → more docs
   - Adaptive parameters

4. **Connection Pooling**
   - Reuse LLM connections
   - Vector store pooling

## Security Features

1. **Input Validation**
   - Sanitization
   - Length limits
   - Character validation

2. **Prompt Injection Protection**
   - Pattern matching
   - Anomaly detection
   - Strict mode option

3. **Rate Limiting**
   - Per-user limits
   - Time windows
   - Graceful degradation

4. **Data Privacy**
   - Session isolation
   - No data persistence (optional)
   - Secure API key handling

## Scalability Considerations

**Current Setup:** Single instance

**Scaling Options:**

1. **Horizontal Scaling**
   - Load balancer + multiple instances
   - Shared ChromaDB cluster
   - Redis for distributed caching

2. **Vertical Scaling**
   - Larger embedding models
   - GPU acceleration
   - Increased batch sizes

3. **Database Scaling**
   - ChromaDB sharding
   - Read replicas
   - Separate read/write instances

## Technology Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | LangGraph | Workflow management |
| LLM Framework | LangChain | Chain building |
| LLM Provider | Anthropic Claude | Answer generation |
| Vector DB | ChromaDB | Document storage |
| Embeddings | SentenceTransformers | Semantic search |
| Reranking | CrossEncoder | Result refinement |
| UI | Streamlit | Web interface |
| Caching | In-memory (TTL) | Performance |
| Monitoring | LangSmith | Observability |
| Testing | Pytest | Quality assurance |

## Future Enhancements

1. **Multi-modal Support**
   - Image understanding in PDFs
   - Vision-language models

2. **Advanced RAG**
   - Graph-based retrieval
   - Hierarchical indexing

3. **Fine-tuning**
   - Domain-specific models
   - Preference learning

4. **Collaboration**
   - Multi-user sessions
   - Shared knowledge base