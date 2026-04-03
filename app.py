import streamlit as st
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
* { font-family: 'DM Sans', sans-serif !important; }
.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 20%, rgba(20,184,166,0.14) 0%, transparent 55%),
        radial-gradient(ellipse 70% 60% at 50% 90%, rgba(139,92,246,0.12) 0%, transparent 60%),
        #0a0f1e !important;
}
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 2.5rem 1.5rem !important; max-width: 760px !important; }
.stMarkdown p, .stMarkdown li { color: #94a3b8 !important; }
h1, h2, h3 { color: #f1f5f9 !important; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    color: #f1f5f9 !important;
    padding: 14px 18px !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(139,92,246,0.6) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
    background: rgba(255,255,255,0.09) !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color: rgba(148,163,184,0.5) !important; }
.stTextInput label, .stTextArea label {
    color: #64748b !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important; border: none !important;
    border-radius: 14px !important; padding: 10px 28px !important;
    font-size: 14px !important; font-weight: 500 !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(99,102,241,0.6) !important;
    transform: translateY(-1px) !important;
}
.stAlert {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #94a3b8 !important;
}
.stSpinner > div { border-top-color: #8b5cf6 !important; }
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #94a3b8 !important; font-size: 13px !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 0 0 14px 14px !important; padding: 16px !important;
}
hr { border-color: rgba(255,255,255,0.07) !important; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important; padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important; font-size: 24px !important; font-weight: 300 !important;
}
.answer-box {
    background: rgba(20,184,166,0.07);
    border: 1px solid rgba(20,184,166,0.22);
    border-radius: 20px; padding: 24px 28px;
    color: #e2e8f0; font-size: 15px; line-height: 1.8; margin: 8px 0 16px;
}
.confidence-bar-wrap {
    margin-top: 12px;
    display: flex; align-items: center; gap: 10px;
}
.confidence-label { color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing:.05em; }
.confidence-bar-bg {
    flex: 1; height: 5px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px; overflow: hidden;
}
.confidence-bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #34d399);
    transition: width 0.6s ease;
}
.confidence-pct { color: #34d399; font-size: 12px; font-weight: 500; min-width: 38px; text-align:right; }
.ctx-card {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.18);
    border-left: 3px solid rgba(139,92,246,0.65);
    border-radius: 14px; padding: 16px 20px;
    color: #94a3b8; font-size: 13px; line-height: 1.7; margin-bottom: 10px;
}
.ctx-label {
    color: #a78bfa; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
}
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(139,92,246,0.18); border: 1px solid rgba(139,92,246,0.3);
    color: #c4b5fd; font-size: 11px; font-weight: 500;
    letter-spacing: 0.05em; text-transform: uppercase;
    padding: 4px 12px; border-radius: 999px; margin-bottom: 12px;
}
.pulse {
    width: 6px; height: 6px; background: #34d399;
    border-radius: 50%; display: inline-block; animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0   rgba(52,211,153,.5); }
    70%  { box-shadow: 0 0 0 5px rgba(52,211,153,0); }
    100% { box-shadow: 0 0 0 0   rgba(52,211,153,0); }
}
.grad-title {
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-size: 2.6rem; font-weight: 300; line-height: 1.2; margin: 0 0 8px;
}
.no-answer-box {
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 16px; padding: 18px 22px;
    color: #fca5a5; font-size: 14px; line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FAISS_PATH   = "faiss_index.bin"
ANSWERS_PATH = "answers.pkl"
EMBED_MODEL  = "all-MiniLM-L6-v2"
QA_MODEL     = "deepset/roberta-base-squad2"   # extractive QA — no hallucination
CONFIDENCE_THRESHOLD = 0.15                    # below this = "not found"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="badge"><span class="pulse"></span> System Ready</div>', unsafe_allow_html=True)
st.markdown('<p class="grad-title">Medical AI Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#64748b;font-size:14px;margin-top:-4px;margin-bottom:28px;">'
    'Extractive QA over <strong style="color:#94a3b8;">500 PubMed QA</strong> records · '
    'FAISS · RoBERTa</p>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)
c1.metric("PubMed Records", "500")
c2.metric("Embedding Dims", "384")
c3.metric("QA Model", "RoBERTa")
st.markdown("<br>", unsafe_allow_html=True)


# ── Load system ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    embed_model = SentenceTransformer(EMBED_MODEL)

    if not os.path.exists(FAISS_PATH) or not os.path.exists(ANSWERS_PATH):
        st.info("⚙️ First-time setup — building knowledge base…")
        prog = st.progress(0, text="Loading dataset…")
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:500]")
        answers = [item['long_answer'] for item in dataset]
        prog.progress(25, text="Creating embeddings…")
        embeddings = embed_model.encode(answers, show_progress_bar=False)
        prog.progress(65, text="Building FAISS index…")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        faiss.write_index(index, FAISS_PATH)
        with open(ANSWERS_PATH, "wb") as f:
            pickle.dump(answers, f)
        prog.progress(100, text="Done!")
    else:
        index = faiss.read_index(FAISS_PATH)
        with open(ANSWERS_PATH, "rb") as f:
            answers = pickle.load(f)

    # Load extractive QA model directly — bypasses broken pipeline task registry
    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    qa_model     = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
    return embed_model, index, answers, qa_tokenizer, qa_model


with st.spinner("Loading AI system…"):
    try:
        embed_model, index, answers, qa_tokenizer, qa_model = load_system()
        loaded = True
    except Exception as e:
        st.error(f"❌ Failed to load: {e}")
        loaded = False

if not loaded:
    st.stop()


# ── Retrieval + QA ─────────────────────────────────────────────────────────────
def retrieve(query, k=5):
    vec = embed_model.encode([query])
    _, idxs = index.search(np.array(vec), k)
    return [answers[i] for i in idxs[0]]

def answer_question(query, retrieved_chunks):
    """
    Extractive QA using AutoModelForQuestionAnswering directly.
    RoBERTa reads context and extracts the exact answer span — no hallucination.
    Returns (answer_text, confidence_score, combined_context).
    """
    combined_context = " ".join(retrieved_chunks)

    inputs = qa_tokenizer(
        query,
        combined_context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = qa_model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end   = torch.argmax(outputs.end_logits) + 1

    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )

    # Confidence = average of softmax-normalised start & end scores
    start_score = torch.softmax(outputs.start_logits, dim=1)[0][start].item()
    end_score   = torch.softmax(outputs.end_logits,   dim=1)[0][end-1].item()
    score = (start_score + end_score) / 2

    return answer, score, combined_context


# ── Query UI ───────────────────────────────────────────────────────────────────
query = st.text_area(
    "ASK A MEDICAL QUESTION",
    placeholder="e.g. What is the universal blood donor group? What causes hypertension?",
    height=110
)

col_btn, col_hint = st.columns([2, 5])
with col_btn:
    ask = st.button("🔍  Search", use_container_width=True)
with col_hint:
    st.markdown(
        '<p style="color:#334155;font-size:12px;padding-top:13px;">Ctrl+Enter to submit</p>',
        unsafe_allow_html=True
    )


# ── Answer ─────────────────────────────────────────────────────────────────────
if ask:
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Searching PubMed knowledge base…"):
            retrieved  = retrieve(query, k=5)
            ans, score, ctx = answer_question(query, retrieved)

        if score < CONFIDENCE_THRESHOLD:
            st.markdown(
                '<div class="no-answer-box">'
                '🔍 The knowledge base doesn\'t contain a confident answer for this question. '
                'Try rephrasing, or this topic may not be covered in the 500 PubMed samples.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            confidence_pct = round(score * 100)
            st.markdown(
                '<p style="color:#34d399;font-size:11px;font-weight:600;'
                'text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">✦ Extracted Answer</p>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="answer-box">'
                f'{ans}'
                f'<div class="confidence-bar-wrap">'
                f'  <span class="confidence-label">Confidence</span>'
                f'  <div class="confidence-bar-bg">'
                f'    <div class="confidence-bar-fill" style="width:{confidence_pct}%"></div>'
                f'  </div>'
                f'  <span class="confidence-pct">{confidence_pct}%</span>'
                f'</div></div>',
                unsafe_allow_html=True
            )

        with st.expander("📚 Retrieved Context — Top 5 PubMed Matches"):
            for i, r in enumerate(retrieved, 1):
                st.markdown(
                    f'<div class="ctx-card">'
                    f'<div class="ctx-label">Match {i} · PubMed QA</div>{r}'
                    f'</div>',
                    unsafe_allow_html=True
                )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p style="color:#1e293b;font-size:11px;text-align:center;">'
    '⚠️ For informational purposes only — not a substitute for professional medical advice.'
    '</p>',
    unsafe_allow_html=True
)
