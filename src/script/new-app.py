# """
# Insurance Chatbot - Streamlit Test Frontend
# Tests: Live Suggestions (typeahead) + Chat

# Run with:
#     streamlit run streamlit_app.py

# Make sure your FastAPI backend is running at http://localhost:8000
# """

# import time
# import uuid
# import requests
# import streamlit as st

# # ─── CONFIG ──────────────────────────────────────────────────────────────────
# API_BASE = "http://localhost:8000"
# PRODUCT  = "zucora"
# DEBOUNCE_SECONDS = 0.35   # wait this long after last keystroke before calling /suggest

# # ─── PAGE SETUP ──────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Zucora Insurance Assistant",
#     page_icon="🛡️",
#     layout="wide",
# )

# # ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

#     html, body, [class*="css"] {
#         font-family: 'DM Sans', sans-serif;
#     }

#     /* ── Page background ── */
#     .stApp {
#         background: #0d0f14;
#         color: #e8e4dc;
#     }

#     /* ── Hide default streamlit chrome ── */
#     #MainMenu, footer, header { visibility: hidden; }

#     /* ── Top banner ── */
#     .banner {
#         background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 60%, #12181f 100%);
#         border-bottom: 1px solid #1e2a3a;
#         padding: 24px 40px 20px;
#         margin: -60px -60px 40px;
#         display: flex;
#         align-items: center;
#         gap: 16px;
#     }
#     .banner-icon { font-size: 2rem; }
#     .banner-title {
#         font-family: 'Syne', sans-serif;
#         font-size: 1.6rem;
#         font-weight: 800;
#         color: #e8e4dc;
#         letter-spacing: -0.5px;
#         margin: 0;
#     }
#     .banner-sub {
#         font-size: 0.78rem;
#         color: #5a6a7a;
#         letter-spacing: 0.08em;
#         text-transform: uppercase;
#         margin: 0;
#     }
#     .badge {
#         margin-left: auto;
#         background: #0e3a2f;
#         border: 1px solid #1a6b55;
#         color: #2ecc9a;
#         font-size: 0.7rem;
#         font-weight: 600;
#         padding: 4px 12px;
#         border-radius: 20px;
#         letter-spacing: 0.06em;
#         text-transform: uppercase;
#     }

#     /* ── Section headers ── */
#     .section-label {
#         font-family: 'Syne', sans-serif;
#         font-size: 0.7rem;
#         font-weight: 700;
#         letter-spacing: 0.14em;
#         text-transform: uppercase;
#         color: #3d8bff;
#         margin-bottom: 10px;
#     }

#     /* ── Suggestion card ── */
#     .suggestion-card {
#         background: #13181f;
#         border: 1px solid #1e2a3a;
#         border-radius: 10px;
#         padding: 12px 16px;
#         margin-bottom: 8px;
#         cursor: pointer;
#         transition: all 0.18s ease;
#         position: relative;
#         overflow: hidden;
#     }
#     .suggestion-card:hover {
#         border-color: #3d8bff;
#         background: #151c28;
#         transform: translateX(3px);
#     }
#     .suggestion-card::before {
#         content: '';
#         position: absolute;
#         left: 0; top: 0; bottom: 0;
#         width: 3px;
#         border-radius: 3px 0 0 3px;
#     }
#     .suggestion-card.keyword::before  { background: #f0a500; }
#     .suggestion-card.semantic::before { background: #3d8bff; }
#     .suggestion-card.both::before     {
#         background: linear-gradient(to bottom, #f0a500, #3d8bff);
#     }
#     .suggestion-question {
#         font-size: 0.9rem;
#         color: #d4cfc8;
#         font-weight: 400;
#         line-height: 1.4;
#         margin: 0 0 6px;
#     }
#     .suggestion-meta {
#         display: flex;
#         align-items: center;
#         gap: 8px;
#     }
#     .tag {
#         font-size: 0.65rem;
#         font-weight: 600;
#         padding: 2px 8px;
#         border-radius: 10px;
#         text-transform: uppercase;
#         letter-spacing: 0.06em;
#     }
#     .tag-keyword  { background: #2a1d00; color: #f0a500; border: 1px solid #3d2800; }
#     .tag-semantic { background: #0a1a33; color: #3d8bff; border: 1px solid #0d2448; }
#     .tag-both     { background: #1a1228; color: #a78bfa; border: 1px solid #2a1a40; }
#     .topic-label {
#         font-size: 0.68rem;
#         color: #3d4a5a;
#         font-style: italic;
#     }
#     .score-bar-wrap {
#         margin-left: auto;
#         display: flex;
#         align-items: center;
#         gap: 6px;
#     }
#     .score-text {
#         font-size: 0.65rem;
#         color: #3d4a5a;
#         font-variant-numeric: tabular-nums;
#     }
#     .score-bar {
#         width: 40px;
#         height: 3px;
#         background: #1e2a3a;
#         border-radius: 2px;
#         overflow: hidden;
#     }
#     .score-fill {
#         height: 100%;
#         border-radius: 2px;
#         background: linear-gradient(to right, #3d8bff, #2ecc9a);
#     }

#     /* ── Chat messages ── */
#     .chat-wrap {
#         display: flex;
#         flex-direction: column;
#         gap: 14px;
#     }
#     .msg-user {
#         display: flex;
#         justify-content: flex-end;
#     }
#     .msg-bot {
#         display: flex;
#         justify-content: flex-start;
#     }
#     .bubble {
#         max-width: 82%;
#         padding: 12px 16px;
#         border-radius: 14px;
#         font-size: 0.88rem;
#         line-height: 1.55;
#     }
#     .bubble-user {
#         background: #1a2d4a;
#         border: 1px solid #1e3a5f;
#         color: #a8c8f0;
#         border-bottom-right-radius: 4px;
#     }
#     .bubble-bot {
#         background: #13181f;
#         border: 1px solid #1e2a3a;
#         color: #d4cfc8;
#         border-bottom-left-radius: 4px;
#     }
#     .bubble-source {
#         font-size: 0.62rem;
#         letter-spacing: 0.08em;
#         text-transform: uppercase;
#         margin-bottom: 6px;
#         font-weight: 600;
#     }
#     .source-faq { color: #2ecc9a; }
#     .source-rag { color: #a78bfa; }

#     /* ── Answer blocks (structured FAQ) ── */
#     .answer-block {
#         background: #0d1117;
#         border: 1px solid #1e2a3a;
#         border-radius: 8px;
#         padding: 10px 14px;
#         margin-top: 8px;
#         font-size: 0.85rem;
#         color: #b8b3ac;
#     }

#     /* ── Empty state ── */
#     .empty-state {
#         text-align: center;
#         padding: 40px 20px;
#         color: #2a3545;
#     }
#     .empty-icon { font-size: 2.5rem; margin-bottom: 10px; }
#     .empty-text { font-size: 0.85rem; }

#     /* ── Status pill ── */
#     .status-pill {
#         display: inline-flex;
#         align-items: center;
#         gap: 6px;
#         font-size: 0.7rem;
#         padding: 4px 10px;
#         border-radius: 20px;
#         margin-bottom: 16px;
#     }
#     .status-ok   { background:#0e3a2f; color:#2ecc9a; border:1px solid #1a6b55; }
#     .status-fail { background:#2a1010; color:#ff6b6b; border:1px solid #5a1a1a; }
#     .status-dot  { width:6px; height:6px; border-radius:50%; background:currentColor; }

#     /* ── Divider ── */
#     .vdivider {
#         width: 1px;
#         background: #1e2a3a;
#         margin: 0 8px;
#     }

#     /* ── Input override ── */
#     .stTextInput > div > div > input {
#         background: #13181f !important;
#         border: 1px solid #1e2a3a !important;
#         border-radius: 10px !important;
#         color: #e8e4dc !important;
#         font-family: 'DM Sans', sans-serif !important;
#         font-size: 0.92rem !important;
#     }
#     .stTextInput > div > div > input:focus {
#         border-color: #3d8bff !important;
#         box-shadow: 0 0 0 2px rgba(61,139,255,0.12) !important;
#     }
#     .stButton > button {
#         background: linear-gradient(135deg, #1a3a6a, #0e2248) !important;
#         border: 1px solid #2a4a8a !important;
#         color: #7ab0ff !important;
#         border-radius: 10px !important;
#         font-family: 'Syne', sans-serif !important;
#         font-weight: 600 !important;
#         letter-spacing: 0.04em !important;
#         transition: all 0.2s !important;
#     }
#     .stButton > button:hover {
#         border-color: #3d8bff !important;
#         color: #a8d0ff !important;
#         transform: translateY(-1px) !important;
#     }
# </style>
# """, unsafe_allow_html=True)


# # ─── SESSION STATE INIT ───────────────────────────────────────────────────────
# if "session_id"      not in st.session_state: st.session_state.session_id      = str(uuid.uuid4())
# if "messages"        not in st.session_state: st.session_state.messages        = []
# if "last_typed"      not in st.session_state: st.session_state.last_typed      = ""
# if "suggestions"     not in st.session_state: st.session_state.suggestions     = []
# if "api_ok"          not in st.session_state: st.session_state.api_ok          = None
# if "chat_input_val"  not in st.session_state: st.session_state.chat_input_val  = ""
# if "clicked_suggest" not in st.session_state: st.session_state.clicked_suggest = None


# # ─── HELPERS ─────────────────────────────────────────────────────────────────
# def check_api():
#     try:
#         r = requests.get(f"{API_BASE}/", timeout=3)
#         st.session_state.api_ok = r.status_code == 200
#     except Exception:
#         st.session_state.api_ok = False


# def fetch_suggestions(query: str) -> list:
#     if len(query.strip()) < 2:
#         return []
#     try:
#         r = requests.post(
#             f"{API_BASE}/suggest",
#             json={"partial_query": query, "product": PRODUCT},
#             timeout=5,
#         )
#         if r.status_code == 200:
#             return r.json().get("suggestions", [])
#     except Exception:
#         pass
#     return []


# def send_chat(query: str) -> dict:
#     try:
#         r = requests.post(
#             f"{API_BASE}/chat",
#             json={
#                 "session_id": st.session_state.session_id,
#                 "product":    PRODUCT,
#                 "user_query": query,
#             },
#             timeout=15,
#         )
#         if r.status_code == 200:
#             return r.json()
#     except Exception as e:
#         return {"source": "error", "answer": str(e)}
#     return {"source": "error", "answer": "Request failed."}


# def render_answer_blocks(blocks):
#     if not blocks:
#         return ""
#     parts = []
#     for b in blocks:
#         btype = b.get("type", "text")
#         if btype == "text":
#             parts.append(f'<div class="answer-block">{b.get("content","")}</div>')
#         elif btype == "list":
#             items = "".join(f"<li>{i}</li>" for i in b.get("items", []))
#             parts.append(f'<div class="answer-block"><ul style="margin:0;padding-left:18px">{items}</ul></div>')
#         elif btype == "link":
#             parts.append(f'<div class="answer-block">🔗 <a href="{b.get("url","#")}" target="_blank" style="color:#3d8bff">{b.get("label", b.get("url",""))}</a></div>')
#         else:
#             parts.append(f'<div class="answer-block">{b}</div>')
#     return "".join(parts)


# # ─── BANNER ──────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="banner">
#     <div class="banner-icon">🛡️</div>
#     <div>
#         <p class="banner-title">Zucora Assistant</p>
#         <p class="banner-sub">Insurance Intelligence · Dev Test Console</p>
#     </div>
#     <div class="badge">⚡ Live API</div>
# </div>
# """, unsafe_allow_html=True)


# # ─── API STATUS ──────────────────────────────────────────────────────────────
# if st.session_state.api_ok is None:
#     check_api()

# col_status, col_refresh = st.columns([5, 1])
# with col_status:
#     if st.session_state.api_ok:
#         st.markdown(f'<div class="status-pill status-ok"><span class="status-dot"></span> API connected · {API_BASE}</div>', unsafe_allow_html=True)
#     else:
#         st.markdown(f'<div class="status-pill status-fail"><span class="status-dot"></span> Cannot reach {API_BASE} — is FastAPI running?</div>', unsafe_allow_html=True)
# with col_refresh:
#     if st.button("↻ Ping", use_container_width=True):
#         check_api()
#         st.rerun()


# # ─── TWO-COLUMN LAYOUT ───────────────────────────────────────────────────────
# left, gap, right = st.columns([5, 0.1, 6])

# with gap:
#     st.markdown('<div class="vdivider" style="height:100vh"></div>', unsafe_allow_html=True)


# # ═══════════════════════════════════════════════════════════
# #  LEFT — SUGGESTION TESTER
# # ═══════════════════════════════════════════════════════════
# with left:
#     st.markdown('<div class="section-label">🔍 Live Suggestions — /suggest</div>', unsafe_allow_html=True)
#     st.caption("Type below to see real-time typeahead suggestions from your knowledge base.")

#     typed = st.text_input(
#         label="Type your question",
#         placeholder="e.g. how do I raise a claim...",
#         key="suggest_input",
#         label_visibility="collapsed",
#     )

#     # If a suggestion was clicked, push it to chat automatically
#     if st.session_state.clicked_suggest:
#         query_to_send = st.session_state.clicked_suggest
#         st.session_state.clicked_suggest = None
#         st.session_state.chat_input_val = query_to_send

#     # Debounced suggestion fetch
#     if typed != st.session_state.last_typed:
#         st.session_state.last_typed = typed
#         if len(typed.strip()) >= 2:
#             time.sleep(DEBOUNCE_SECONDS)
#             st.session_state.suggestions = fetch_suggestions(typed)
#         else:
#             st.session_state.suggestions = []

#     # Render suggestions
#     suggestions = st.session_state.suggestions

#     if suggestions:
#         st.markdown(f'<div style="font-size:0.72rem;color:#3d4a5a;margin-bottom:10px">{len(suggestions)} suggestion{"s" if len(suggestions)>1 else ""} found</div>', unsafe_allow_html=True)

#         for i, s in enumerate(suggestions):
#             mtype   = s.get("match_type", "semantic")
#             score   = s.get("score", 0.0)
#             pct     = int(score * 100)
#             topic   = s.get("topic_name") or s.get("topic_id", "")
#             q_text  = s.get("question", "")

#             tag_class = {"keyword": "tag-keyword", "semantic": "tag-semantic", "both": "tag-both"}.get(mtype, "tag-semantic")
#             card_class = f"suggestion-card {mtype}"

#             st.markdown(f"""
#             <div class="{card_class}">
#                 <p class="suggestion-question">{q_text}</p>
#                 <div class="suggestion-meta">
#                     <span class="tag {tag_class}">{mtype}</span>
#                     <span class="topic-label">{topic}</span>
#                     <div class="score-bar-wrap">
#                         <span class="score-text">{pct}%</span>
#                         <div class="score-bar">
#                             <div class="score-fill" style="width:{pct}%"></div>
#                         </div>
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#             if st.button(f"Send to chat ↗", key=f"use_suggestion_{i}", use_container_width=True):
#                 st.session_state.clicked_suggest = q_text
#                 st.rerun()

#     elif typed and len(typed.strip()) >= 2:
#         st.markdown("""
#         <div class="empty-state">
#             <div class="empty-icon">🔭</div>
#             <div class="empty-text">No suggestions matched.<br>Try different keywords or check the API.</div>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown("""
#         <div class="empty-state">
#             <div class="empty-icon">💬</div>
#             <div class="empty-text">Start typing to see live suggestions appear here.</div>
#         </div>
#         """, unsafe_allow_html=True)

#     # ── Raw JSON toggle ──
#     if suggestions:
#         with st.expander("🔩 Raw API response (JSON)"):
#             st.json(suggestions)


# # ═══════════════════════════════════════════════════════════
# #  RIGHT — CHAT
# # ═══════════════════════════════════════════════════════════
# with right:
#     st.markdown('<div class="section-label">💬 Chat — /chat</div>', unsafe_allow_html=True)
#     st.caption(f"Session: `{st.session_state.session_id[:18]}…`")

#     # ── Message history ──
#     chat_container = st.container(height=460)
#     with chat_container:
#         if not st.session_state.messages:
#             st.markdown("""
#             <div class="empty-state" style="padding-top:80px">
#                 <div class="empty-icon">🛡️</div>
#                 <div class="empty-text">Ask anything about your insurance.<br>Click a suggestion on the left to auto-fill.</div>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
#             for msg in st.session_state.messages:
#                 if msg["role"] == "user":
#                     st.markdown(f"""
#                     <div class="msg-user">
#                         <div class="bubble bubble-user">{msg["content"]}</div>
#                     </div>""", unsafe_allow_html=True)
#                 else:
#                     source = msg.get("source", "rag")
#                     source_label = "📋 Structured FAQ" if source == "structured_faq" else "🤖 RAG"
#                     source_cls   = "source-faq"         if source == "structured_faq" else "source-rag"

#                     if source == "structured_faq" and msg.get("answer_blocks"):
#                         body = render_answer_blocks(msg["answer_blocks"])
#                     else:
#                         body = f'<p style="margin:0">{msg.get("content","")}</p>'

#                     st.markdown(f"""
#                     <div class="msg-bot">
#                         <div class="bubble bubble-bot">
#                             <div class="bubble-source {source_cls}">{source_label}</div>
#                             {body}
#                         </div>
#                     </div>""", unsafe_allow_html=True)
#             st.markdown('</div>', unsafe_allow_html=True)

#     # ── Input row ──
#     chat_col1, chat_col2 = st.columns([8, 2])

#     with chat_col1:
#         # Pre-fill if a suggestion was clicked
#         prefill = st.session_state.get("chat_input_val", "")
#         user_input = st.text_input(
#             label="chat_input",
#             value=prefill,
#             placeholder="Ask a question or click a suggestion →",
#             key="chat_input",
#             label_visibility="collapsed",
#         )
#         if prefill:
#             st.session_state.chat_input_val = ""  # clear after render

#     with chat_col2:
#         send = st.button("Send ↗", use_container_width=True)

#     if send and user_input and user_input.strip():
#         query = user_input.strip()

#         # Add user message
#         st.session_state.messages.append({"role": "user", "content": query})

#         # Call API
#         with st.spinner("Thinking..."):
#             result = send_chat(query)

#         # Add assistant message
#         assistant_msg = {
#             "role":    "assistant",
#             "source":  result.get("source", "rag"),
#             "content": result.get("answer", ""),
#             "answer_blocks": result.get("answer_blocks"),
#         }
#         st.session_state.messages.append(assistant_msg)
#         st.rerun()

#     # ── Controls row ──
#     ctrl1, ctrl2 = st.columns(2)
#     with ctrl1:
#         if st.button("🗑 Clear chat", use_container_width=True):
#             st.session_state.messages = []
#             st.rerun()
#     with ctrl2:
#         if st.button("🔄 New session", use_container_width=True):
#             st.session_state.session_id = str(uuid.uuid4())
#             st.session_state.messages   = []
#             st.rerun()

#     # ── Last response raw JSON ──
#     if st.session_state.messages:
#         last_bot = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
#         if last_bot:
#             with st.expander("🔩 Raw last response (JSON)"):
#                 st.json(last_bot)


"""
Zucora Insurance Assistant - Streamlit UI
Fixes:
  1. Dropdown no longer hides the Ask button (buttons are INSIDE the search component)
  2. Reliable Streamlit bridge via st.query_params (no fragile DOM hacking)

Run with:
    pip install streamlit requests
    streamlit run streamlit_app.py

FastAPI backend must be running at http://localhost:8000
"""

import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000"
PRODUCT  = "zucora"

# ─── PAGE ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Zucora Assistant", page_icon="🛡️", layout="centered")

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if "messages"   not in st.session_state: st.session_state.messages   = []

# ─── READ QUERY PARAM (bridge from JS) ───────────────────────────────────────
# JS sets ?q=<user query> in the iframe URL → parent page detects it via
# st.query_params which triggers a rerun automatically in Streamlit.
params = st.query_params
incoming = params.get("q", "").strip()

if incoming:
    st.query_params.clear()          # consume immediately to avoid re-processing

    if incoming == "__CLEAR__":
        st.session_state.messages = []
        st.rerun()

    elif incoming == "__NEW_SESSION__":
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages   = []
        st.rerun()

    else:
        # Real query → call chat API
        st.session_state.messages.append({"role": "user", "content": incoming})
        try:
            r = requests.post(
                f"{API_BASE}/chat",
                json={"session_id": st.session_state.session_id,
                      "product": PRODUCT, "user_query": incoming},
                timeout=15,
            )
            result = r.json() if r.status_code == 200 else {"source": "error", "answer": f"HTTP {r.status_code}"}
        except Exception as e:
            result = {"source": "error", "answer": str(e)}

        st.session_state.messages.append({
            "role": "assistant",
            "source": result.get("source", "rag"),
            "content": result.get("answer", ""),
            "answer_blocks": result.get("answer_blocks"),
        })
        st.rerun()

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def render_blocks(blocks):
    if not blocks: return ""
    out = []
    for b in blocks:
        t = b.get("type", "text")
        if t == "text":
            out.append(f'<div class="ab">{b.get("content","")}</div>')
        elif t == "list":
            li = "".join(f"<li>{i}</li>" for i in b.get("items", []))
            out.append(f'<div class="ab"><ul style="margin:4px 0;padding-left:16px">{li}</ul></div>')
        elif t == "link":
            out.append(f'<div class="ab">🔗 <a href="{b.get("url","#")}" target="_blank" style="color:#1a73e8">{b.get("label","Link")}</a></div>')
    return "".join(out)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#f1f3f4;}
#MainMenu,footer,header{visibility:hidden;}
.main .block-container{max-width:680px;padding-top:44px;padding-bottom:32px;}

.logo{text-align:center;margin-bottom:28px;}
.logo h1{font-size:1.75rem;font-weight:600;color:#202124;letter-spacing:-.4px;margin:0;}
.logo p{font-size:.84rem;color:#5f6368;margin:5px 0 0;}

.chat-area{margin-bottom:16px;}
.mu{display:flex;justify-content:flex-end;margin-bottom:9px;}
.mb{display:flex;justify-content:flex-start;margin-bottom:9px;}
.bbl{max-width:76%;padding:11px 16px;border-radius:18px;font-size:.875rem;line-height:1.55;}
.bbl-u{background:#1a73e8;color:#fff;border-bottom-right-radius:4px;}
.bbl-b{background:#fff;border:1px solid #e0e0e0;color:#202124;border-bottom-left-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.07);}
.slbl{font-size:.6rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;opacity:.55;margin-bottom:5px;}
.ab{padding:5px 0;border-top:1px solid #f1f3f4;font-size:.84rem;}
.ab:first-child{border-top:none;}

.foot{text-align:center;margin-top:12px;font-size:.7rem;color:#bdc1c6;}
</style>
""", unsafe_allow_html=True)

# ─── LOGO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="logo">
  <h1>🛡️ Zucora Assistant</h1>
  <p>Ask anything about your insurance coverage</p>
</div>
""", unsafe_allow_html=True)

# ─── CHAT HISTORY ────────────────────────────────────────────────────────────
if st.session_state.messages:
    parts = ['<div class="chat-area">']
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            parts.append(f'<div class="mu"><div class="bbl bbl-u">{msg["content"]}</div></div>')
        else:
            src   = msg.get("source","rag")
            lbl   = "📋 Structured FAQ" if src == "structured_faq" else "🤖 AI Answer"
            color = "#1e8c6e"           if src == "structured_faq" else "#7c4dff"
            body  = render_blocks(msg["answer_blocks"]) if src == "structured_faq" and msg.get("answer_blocks") \
                    else f'<p style="margin:0">{msg.get("content","")}</p>'
            parts.append(f'''<div class="mb"><div class="bbl bbl-b">
                <div class="slbl" style="color:{color}">{lbl}</div>{body}
            </div></div>''')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)

# ─── SEARCH COMPONENT ────────────────────────────────────────────────────────
# The component lives inside a fixed-height iframe.
# It talks back to the parent page by setting window.parent.location to
# ?q=<encoded_query> which Streamlit picks up as a query_param change → rerun.

COMPONENT_HEIGHT = 260   # tall enough to show suggestions + buttons without scroll

search_html = f"""
<!DOCTYPE html><html>
<head>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Inter',-apple-system,sans-serif;background:transparent;padding:0 2px 4px;}}

.wrap{{position:relative;width:100%;}}

/* ── Search bar ── */
.sbox{{
    display:flex;align-items:center;gap:10px;
    background:#fff;border:1px solid #dfe1e5;border-radius:24px;
    padding:13px 18px;
    box-shadow:0 1px 6px rgba(32,33,36,.10);
    transition:box-shadow .2s,border-radius .15s;
}}
.sbox:focus-within{{box-shadow:0 2px 10px rgba(32,33,36,.18);}}
.sbox.open{{border-radius:24px 24px 0 0;border-bottom-color:#e8eaed;}}

.ico{{color:#9aa0a6;font-size:.95rem;flex-shrink:0;}}
#q{{
    flex:1;border:none;outline:none;
    font-size:1rem;font-family:inherit;color:#202124;
    background:transparent;caret-color:#1a73e8;min-width:0;
}}
#q::placeholder{{color:#9aa0a6;}}
.xbtn{{
    background:none;border:none;cursor:pointer;
    color:#9aa0a6;font-size:.9rem;padding:0 2px;
    display:none;flex-shrink:0;line-height:1;
}}
.xbtn:hover{{color:#5f6368;}}

/* ── Dropdown — absolutely positioned, does NOT push buttons down ── */
#dd{{
    position:absolute;   /* key: taken out of normal flow */
    top:100%;left:0;right:0;
    background:#fff;
    border:1px solid #dfe1e5;border-top:1px solid #e8eaed;
    border-radius:0 0 20px 20px;
    box-shadow:0 6px 16px rgba(32,33,36,.15);
    overflow:hidden;
    display:none;
    z-index:999;         /* floats above buttons */
    max-height:220px;
    overflow-y:auto;
}}
#dd.vis{{display:block;}}

.si{{
    display:flex;align-items:center;padding:11px 18px;
    cursor:pointer;gap:11px;border-top:1px solid #f1f3f4;
    transition:background .1s;
}}
.si:first-child{{border-top:none;}}
.si:hover{{background:#f8f9fa;}}
.si:hover .sq{{color:#1a73e8;}}
.si:hover .arr{{color:#1a73e8;transform:translateX(3px);}}

.sico{{color:#bdc1c6;font-size:.85rem;flex-shrink:0;}}
.sq{{flex:1;font-size:.9rem;color:#202124;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;transition:color .1s;}}
.sq strong{{font-weight:600;}}
.arr{{color:#c5c8cb;font-size:.8rem;flex-shrink:0;transition:color .15s,transform .15s;}}

.badge{{
    font-size:.58rem;font-weight:600;padding:2px 7px;
    border-radius:10px;text-transform:uppercase;letter-spacing:.06em;flex-shrink:0;
}}
.bkw{{background:#fef3e2;color:#e37400;}}
.bsm{{background:#e8f0fe;color:#1558d6;}}
.bbt{{background:#f3e8fd;color:#7c4dff;}}

.info{{padding:12px 18px;color:#9aa0a6;font-size:.84rem;display:flex;align-items:center;gap:9px;}}
.spin{{width:13px;height:13px;border:2px solid #e0e0e0;border-top-color:#1a73e8;border-radius:50%;animation:sp .7s linear infinite;flex-shrink:0;}}
@keyframes sp{{to{{transform:rotate(360deg)}}}}

/* ── Buttons — always below the bar in normal flow ── */
.btnrow{{
    display:flex;gap:8px;justify-content:center;
    margin-top:14px;      /* gap between bar and buttons */
    position:relative;    /* normal flow — not pushed by absolute dropdown */
    z-index:1;
}}
.bask{{
    background:#1a73e8;color:#fff;border:none;border-radius:20px;
    padding:9px 26px;font-size:.86rem;font-family:inherit;font-weight:500;
    cursor:pointer;transition:background .18s,box-shadow .18s;
}}
.bask:hover{{background:#1557b0;box-shadow:0 2px 8px rgba(26,115,232,.3);}}
.bask:disabled{{background:#c5cae9;cursor:not-allowed;box-shadow:none;}}
.bsec{{
    background:#fff;color:#5f6368;border:1px solid #dfe1e5;border-radius:20px;
    padding:9px 16px;font-size:.86rem;font-family:inherit;
    cursor:pointer;transition:background .15s;
}}
.bsec:hover{{background:#f8f9fa;}}
</style>
</head>
<body>

<div class="wrap">
  <div class="sbox" id="sbox">
    <span class="ico">🔍</span>
    <input id="q" type="text" placeholder="Ask about your insurance coverage…" autocomplete="off"/>
    <button class="xbtn" id="xbtn" onclick="clearQ()">✕</button>
  </div>
  <div id="dd"></div>   <!-- absolutely positioned, floats above buttons -->
</div>

<!-- These are in normal flow and never hidden by the dropdown -->
<div class="btnrow">
  <button class="bask" id="askbtn" onclick="submitQ()" disabled>Ask Zucora</button>
  <button class="bsec" onclick="send('__CLEAR__')">Clear chat</button>
  <button class="bsec" onclick="send('__NEW_SESSION__')">New session</button>
</div>

<script>
const API  = "{API_BASE}";
const PROD = "{PRODUCT}";
const DBNC = 300;

const qEl    = document.getElementById('q');
const xbtn   = document.getElementById('xbtn');
const dd     = document.getElementById('dd');
const sbox   = document.getElementById('sbox');
const askbtn = document.getElementById('askbtn');

let timer = null, cur = '', hits = [];

qEl.addEventListener('input', () => {{
    cur = qEl.value.trim();
    xbtn.style.display = qEl.value ? 'block' : 'none';
    askbtn.disabled    = !cur;
    if (cur.length < 2) {{ close(); return; }}
    showLoad();
    clearTimeout(timer);
    timer = setTimeout(() => fetchS(cur), DBNC);
}});

qEl.addEventListener('keydown', e => {{
    if (e.key === 'Enter' && cur) submitQ();
    if (e.key === 'Escape') close();
}});

document.addEventListener('click', e => {{
    if (!e.target.closest('.wrap')) close();
}});

async function fetchS(q) {{
    if (q !== cur) return;
    try {{
        const r = await fetch(`${{API}}/suggest`, {{
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{partial_query:q, product:PROD}})
        }});
        const data = await r.json();
        if (q !== cur) return;
        hits = data.suggestions || [];
        renderDD(hits, q);
    }} catch(e) {{ close(); }}
}}

function renderDD(sugg, q) {{
    if (!sugg.length) {{
        dd.innerHTML = '<div class="info">No suggestions found — try different keywords</div>';
        open(); return;
    }}
    dd.innerHTML = sugg.map((s,i) => {{
        const bc = {{keyword:'bkw',semantic:'bsm',both:'bbt'}}[s.match_type]||'bsm';
        const hl = s.question.replace(
            new RegExp('(' + q.replace(/[.*+?^${{}}()|[\\]\\\\]/g,'\\\\$&') + ')','gi'),
            '<strong>$1</strong>'
        );
        return `<div class="si" onclick="pick(${{i}})">
            <span class="sico">🔎</span>
            <span class="sq">${{hl}}</span>
            <span class="badge ${{bc}}">${{s.match_type}}</span>
            <span class="arr">→</span>
        </div>`;
    }}).join('');
    open();
}}

function showLoad() {{
    dd.innerHTML = '<div class="info"><div class="spin"></div>Finding suggestions…</div>';
    open();
}}

function open()  {{ dd.classList.add('vis');    sbox.classList.add('open');    }}
function close() {{ dd.classList.remove('vis'); sbox.classList.remove('open'); }}

function pick(i) {{
    const s = hits[i]; if(!s) return;
    qEl.value = s.question; cur = s.question;
    askbtn.disabled = false; xbtn.style.display = 'block';
    close();
    submitQ();
}}

function clearQ() {{
    qEl.value=''; cur='';
    xbtn.style.display='none'; askbtn.disabled=true;
    close(); qEl.focus();
}}

function submitQ() {{
    if (!cur) return;
    send(cur);
    clearQ();
}}

/* ── Reliable bridge: navigate parent to ?q=<value> ── */
function send(value) {{
    const encoded = encodeURIComponent(value);
    window.parent.location.href = window.parent.location.pathname + '?q=' + encoded;
}}
</script>
</body>
</html>
"""

st.components.v1.html(search_html, height=COMPONENT_HEIGHT, scrolling=False)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
n = len(st.session_state.messages) // 2
st.markdown(f"""
<div class="foot">
  Session: {st.session_state.session_id[:22]}…
  &nbsp;·&nbsp; {n} exchange{"s" if n!=1 else ""}
</div>
""", unsafe_allow_html=True)