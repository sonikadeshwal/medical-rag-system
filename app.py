import streamlit as st
import faiss
import pickle
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset

st.set_page_config(
    page_title="MedQuery AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Nunito:wght@300;400;500;600&display=swap');

/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Force white background everywhere ── */
html, body, .stApp, .main, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stVerticalBlock"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    font-family: 'Nunito', sans-serif !important;
}

/* ── All text defaults ── */
p, span, div, li, label {
    font-family: 'Nunito', sans-serif !important;
    color: #1a1a2e !important;
}

/* ── Inputs: force visible text ── */
textarea, input {
    font-family: 'Nunito', sans-serif !important;
    font-size: 15px !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
    background-color: #f8f9ff !important;
    background: #f8f9ff !important;
    border: 2px solid #e8e8f0 !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    caret-color: #6c63ff !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: none !important;
}
textarea:focus, input:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 4px rgba(108,99,255,0.10) !important;
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
    outline: none !important;
}
textarea::placeholder, input::placeholder {
    color: #aaa !important;
    -webkit-text-fill-color: #aaa !important;
}

/* ── Streamlit label ── */
label, [data-testid="stWidgetLabel"] p {
    display: none !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff 0%, #48cfad 100%) !important;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 12px 36px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.30) !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(108,99,255,0.40) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #6c63ff !important; }

/* ── Expander ── */
[data-testid="stExpander"], details {
    background: #f8f9ff !important;
    background-color: #f8f9ff !important;
    border: 1.5px solid #e8e8f0 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
details summary, [data-testid="stExpander"] summary {
    background: #f8f9ff !important;
    background-color: #f8f9ff !important;
    color: #6c63ff !important;
    -webkit-text-fill-color: #6c63ff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 14px 18px !important;
}
details > div, [data-testid="stExpander"] > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    padding: 16px !important;
}

/* ── Alert ── */
.stAlert {
    border-radius: 14px !important;
    font-family: 'Nunito', sans-serif !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #f8f9ff !important;
    border: 1.5px solid #e8e8f0 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    text-align: center !important;
}
[data-testid="stMetricValue"] {
    color: #6c63ff !important;
    -webkit-text-fill-color: #6c63ff !important;
    font-size: 26px !important;
    font-weight: 600 !important;
    font-family: 'Playfair Display', serif !important;
}
[data-testid="stMetricLabel"] p {
    display: block !important;
    color: #888 !important;
    -webkit-text-fill-color: #888 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f1f1; }
::-webkit-scrollbar-thumb { background: #d0d0e8; border-radius: 99px; }
</style>

<style>
/* ── Animated page wrapper ── */
.page-wrap {
    max-width: 720px;
    margin: 0 auto;
    padding: 48px 24px 60px;
    animation: pageIn 0.6s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes pageIn {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: none; }
}

/* ── Header ── */
.hero {
    text-align: center;
    margin-bottom: 40px;
}
.hero-icon {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #6c63ff, #48cfad);
    border-radius: 20px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 30px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(108,99,255,0.25);
    animation: iconPop 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.2s both;
}
@keyframes iconPop {
    from { opacity: 0; transform: scale(0.5) rotate(-10deg); }
    to   { opacity: 1; transform: scale(1) rotate(0deg); }
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.6rem;
    font-weight: 600;
    color: #1a1a2e !important;
    -webkit-text-fill-color: transparent !important;
    background: linear-gradient(135deg, #1a1a2e 0%, #6c63ff 60%, #48cfad 100%);
    -webkit-background-clip: text;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 10px;
    animation: fadeUp 0.6s ease 0.3s both;
}
.hero-sub {
    color: #888 !important;
    -webkit-text-fill-color: #888 !important;
    font-size: 15px;
    line-height: 1.6;
    animation: fadeUp 0.6s ease 0.4s both;
}
.hero-sub strong {
    color: #6c63ff !important;
    -webkit-text-fill-color: #6c63ff !important;
    font-weight: 600;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: none; }
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: #f0fdf9;
    border: 1.5px solid #bbf7e8;
    border-radius: 99px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #10b981 !important;
    -webkit-text-fill-color: #10b981 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 16px;
    animation: fadeUp 0.6s ease 0.1s both;
}
.live-dot {
    width: 7px; height: 7px;
    background: #10b981;
    border-radius: 50%;
    animation: livePulse 1.8s ease-in-out infinite;
}
@keyframes livePulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    50%       { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}

/* ── Stats ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 32px;
    animation: fadeUp 0.6s ease 0.5s both;
}
.stat-box {
    background: #f8f9ff;
    border: 1.5px solid #e8e8f0;
    border-radius: 16px;
    padding: 20px 12px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(108,99,255,0.10);
}
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 600;
    color: #6c63ff !important;
    -webkit-text-fill-color: #6c63ff !important;
}
.stat-lbl {
    font-size: 11px;
    color: #aaa !important;
    -webkit-text-fill-color: #aaa !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 4px;
}

/* ── Search box wrapper ── */
.search-wrap {
    background: #ffffff;
    border: 2px solid #e8e8f0;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(108,99,255,0.06);
    animation: fadeUp 0.6s ease 0.55s both;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.search-wrap:focus-within {
    border-color: #6c63ff;
    box-shadow: 0 4px 32px rgba(108,99,255,0.14);
}
.search-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #aaa !important;
    -webkit-text-fill-color: #aaa !important;
    margin-bottom: 10px;
}

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, #f0f4ff 0%, #f0fdfb 100%);
    border: 1.5px solid #ddd6fe;
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
    animation: cardIn 0.5s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes cardIn {
    from { opacity: 0; transform: translateY(16px) scale(0.98); }
    to   { opacity: 1; transform: none; }
}
.answer-tag {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6c63ff !important;
    -webkit-text-fill-color: #6c63ff !important;
    margin-bottom: 12px;
    display: flex; align-items: center; gap: 6px;
}
.answer-tag::before {
    content: '';
    width: 20px; height: 2px;
    background: #6c63ff;
    border-radius: 2px;
    display: inline-block;
}
.answer-text {
    font-size: 16px !important;
    line-height: 1.8 !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
    font-weight: 400;
}

/* ── Confidence bar ── */
.conf-row {
    display: flex; align-items: center; gap: 12px;
    margin-top: 18px; padding-top: 18px;
    border-top: 1px solid rgba(108,99,255,0.15);
}
.conf-label {
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: #aaa !important; -webkit-text-fill-color: #aaa !important;
    white-space: nowrap;
}
.conf-track {
    flex: 1; height: 6px;
    background: #e8e8f0; border-radius: 99px; overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #6c63ff, #48cfad);
    transition: width 1s cubic-bezier(0.22,1,0.36,1);
}
.conf-pct {
    font-size: 13px; font-weight: 700;
    color: #6c63ff !important; -webkit-text-fill-color: #6c63ff !important;
    min-width: 40px; text-align: right;
}

/* ── Context cards ── */
.ctx-wrap {
    animation: fadeUp 0.5s ease 0.1s both;
}
.ctx-item {
    background: #fff;
    border: 1.5px solid #e8e8f0;
    border-left: 4px solid #6c63ff;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.ctx-item:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 16px rgba(108,99,255,0.10);
}
.ctx-num {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #6c63ff !important; -webkit-text-fill-color: #6c63ff !important;
    margin-bottom: 6px;
}
.ctx-text {
    font-size: 13px; line-height: 1.7;
    color: #555 !important; -webkit-text-fill-color: #555 !important;
}

/* ── No-answer card ── */
.no-ans {
    background: #fff5f5;
    border: 1.5px solid #fecdd3;
    border-radius: 16px;
    padding: 20px 24px;
    animation: cardIn 0.4s ease both;
}
.no-ans-text {
    color: #e11d48 !important;
    -webkit-text-fill-color: #e11d48 !important;
    font-size: 14px; line-height: 1.7;
}

/* ── Suggestions ── */
.sug-wrap {
    background: #f8f9ff;
    border: 1.5px solid #e8e8f0;
    border-radius: 16px;
    padding: 20px 24px;
    margin-top: 8px;
    animation: fadeUp 0.6s ease 0.6s both;
}
.sug-title {
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #ccc !important; -webkit-text-fill-color: #ccc !important;
    margin-bottom: 12px;
}
.sug-chips {
    display: flex; flex-wrap: wrap; gap: 8px;
}
.sug-chip {
    background: #fff;
    border: 1.5px solid #e0dffe;
    border-radius: 99px;
    padding: 7px 16px;
    font-size: 13px; font-weight: 500;
    color: #6c63ff !important; -webkit-text-fill-color: #6c63ff !important;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Nunito', sans-serif;
}
.sug-chip:hover {
    background: #6c63ff;
    color: #fff !important; -webkit-text-fill-color: #fff !important;
    border-color: #6c63ff;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(108,99,255,0.25);
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #e8e8f0, transparent);
    margin: 28px 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #ccc !important;
    -webkit-text-fill-color: #ccc !important;
    font-size: 12px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ── JS: fix any leftover overrides + animate chips ─────────────────────────────
st.markdown("""
<script>
(function() {
    function applyFixes() {
        /* inputs */
        document.querySelectorAll('textarea, input').forEach(function(el) {
            el.style.setProperty('color',                   '#1a1a2e', 'important');
            el.style.setProperty('-webkit-text-fill-color', '#1a1a2e', 'important');
            el.style.setProperty('background-color',        '#f8f9ff', 'important');
            el.style.setProperty('background',              '#f8f9ff', 'important');
            el.style.setProperty('border',         '2px solid #e8e8f0','important');
            el.style.setProperty('border-radius',           '14px',    'important');
            el.style.setProperty('caret-color',             '#6c63ff', 'important');
            el.style.setProperty('font-family',  'Nunito, sans-serif', 'important');
            el.style.setProperty('font-size',               '15px',    'important');
            el.style.setProperty('padding',          '14px 18px',      'important');
        });
        /* expander */
        document.querySelectorAll('details, details summary').forEach(function(el) {
            el.style.setProperty('background-color', '#f8f9ff', 'important');
            el.style.setProperty('background',       '#f8f9ff', 'important');
            el.style.setProperty('color',            '#6c63ff', 'important');
            el.style.setProperty('-webkit-text-fill-color', '#6c63ff', 'important');
        });
        /* answer text */
        document.querySelectorAll('.answer-text').forEach(function(el) {
            el.style.setProperty('color',                   '#1a1a2e', 'important');
            el.style.setProperty('-webkit-text-fill-color', '#1a1a2e', 'important');
        });
        /* chip click → fill textarea */
        document.querySelectorAll('.sug-chip').forEach(function(chip) {
            chip.onclick = function() {
                var ta = document.querySelector('textarea');
                if (ta) {
                    var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                    nativeInputValueSetter.call(ta, chip.innerText);
                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                    ta.focus();
                }
            };
        });
    }
    applyFixes();
    new MutationObserver(applyFixes).observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FAISS_PATH  = "faiss_index.bin"
ANS_PATH    = "answers.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
QA_MODEL    = "deepset/roberta-base-squad2"
CONF_THRESH = 0.15

# ── Page HTML ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-wrap">
  <div class="hero">
    <div class="status-pill"><span class="live-dot"></span> Live System</div>
    <div class="hero-icon">🩺</div>
    <div class="hero-title">MedQuery AI</div>
    <div class="hero-sub">Extractive answers from <strong>500 PubMed</strong> research papers · Zero hallucination</div>
  </div>
  <div class="stats-row">
    <div class="stat-box"><div class="stat-val">500</div><div class="stat-lbl">PubMed Records</div></div>
    <div class="stat-box"><div class="stat-val">384</div><div class="stat-lbl">Embed Dims</div></div>
    <div class="stat-box"><div class="stat-val">Top 5</div><div class="stat-lbl">Context Chunks</div></div>
  </div>
  <div class="search-label">Ask a Medical Question</div>
</div>
""", unsafe_allow_html=True)

# ── Load system ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    embed_model = SentenceTransformer(EMBED_MODEL)

    if not os.path.exists(FAISS_PATH) or not os.path.exists(ANS_PATH):
        prog = st.progress(0, text="Downloading PubMed QA dataset…")
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train[:500]")
        answers = [item['long_answer'] for item in ds]
        prog.progress(30, text="Creating embeddings…")
        embs = embed_model.encode(answers, show_progress_bar=False)
        prog.progress(65, text="Building FAISS index…")
        idx = faiss.IndexFlatL2(embs.shape[1])
        idx.add(np.array(embs))
        faiss.write_index(idx, FAISS_PATH)
        with open(ANS_PATH, "wb") as f:
            pickle.dump(answers, f)
        prog.progress(100, text="Ready!")
    else:
        idx = faiss.read_index(FAISS_PATH)
        with open(ANS_PATH, "rb") as f:
            answers = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model     = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
    return embed_model, idx, answers, tokenizer, model

with st.spinner("Loading AI system — this takes ~1 min on first run…"):
    try:
        embed_model, idx, answers, tokenizer, model = load_system()
        ready = True
    except Exception as e:
        st.error(f"Failed to load: {e}")
        ready = False

if not ready:
    st.stop()

# ── Helpers ────────────────────────────────────────────────────────────────────
def retrieve(query, k=5):
    vec = embed_model.encode([query])
    _, idxs = idx.search(np.array(vec, dtype="float32"), k)
    return [answers[i] for i in idxs[0]]

def extract_answer(question, chunks):
    context = " ".join(chunks)
    inputs  = tokenizer(question, context, return_tensors="pt",
                        truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**inputs)
    s = torch.argmax(out.start_logits)
    e = torch.argmax(out.end_logits) + 1
    ans = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][s:e])
    )
    sc = (torch.softmax(out.start_logits, dim=1)[0][s].item() +
          torch.softmax(out.end_logits,   dim=1)[0][e-1].item()) / 2
    return ans.strip(), sc, context

# ── Input UI ───────────────────────────────────────────────────────────────────
query = st.text_area(
    "question",
    placeholder="e.g. What is the universal blood donor group? What causes hypertension?",
    height=120,
    label_visibility="collapsed"
)

col1, col2 = st.columns([3, 2])
with col1:
    search = st.button("🔍  Search PubMed", use_container_width=True)

# ── Suggestions ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sug-wrap">
  <div class="sug-title">Try these</div>
  <div class="sug-chips">
    <button class="sug-chip">What is diabetes?</button>
    <button class="sug-chip">What causes hypertension?</button>
    <button class="sug-chip">How does chemotherapy work?</button>
    <button class="sug-chip">What is insulin resistance?</button>
    <button class="sug-chip">What is a myocardial infarction?</button>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Answer ─────────────────────────────────────────────────────────────────────
if search:
    if not query.strip():
        st.warning("Please type a question above.")
    else:
        with st.spinner("Searching and extracting answer…"):
            chunks  = retrieve(query)
            ans, sc, _ = extract_answer(query, chunks)

        pct = round(sc * 100)

        if sc < CONF_THRESH or len(ans.strip()) < 3:
            st.markdown("""
            <div style="background:#fff5f5;border:1.5px solid #fecdd3;border-radius:16px;padding:20px 24px;">
              <p style="color:#e11d48 !important;-webkit-text-fill-color:#e11d48 !important;font-size:14px;line-height:1.7;margin:0;">
                🔍 No confident answer found in the PubMed knowledge base for this question.
                Try rephrasing, or this topic may not be covered in the 500 sample records.
              </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#f0f4ff 0%,#f0fdfb 100%);border:1.5px solid #ddd6fe;border-radius:20px;padding:28px;margin-bottom:20px;animation:cardIn 0.5s cubic-bezier(0.22,1,0.36,1) both;">
              <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#6c63ff !important;-webkit-text-fill-color:#6c63ff !important;margin-bottom:14px;display:flex;align-items:center;gap:8px;">
                <span style="width:20px;height:2px;background:#6c63ff;border-radius:2px;display:inline-block;"></span>
                Extracted Answer
              </div>
              <p style="font-size:16px !important;line-height:1.8 !important;color:#1a1a2e !important;-webkit-text-fill-color:#1a1a2e !important;font-weight:400;margin:0 0 18px 0;">{ans}</p>
              <div style="display:flex;align-items:center;gap:12px;padding-top:18px;border-top:1px solid rgba(108,99,255,0.15);">
                <span style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#aaa !important;-webkit-text-fill-color:#aaa !important;white-space:nowrap;">Confidence</span>
                <div class="conf-track">
                  <div style="flex:1;height:6px;background:#e8e8f0;border-radius:99px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;border-radius:99px;background:linear-gradient(90deg,#6c63ff,#48cfad);transition:width 1s cubic-bezier(0.22,1,0.36,1);"></div>
                  </div>
                <span style="font-size:13px;font-weight:700;color:#6c63ff !important;-webkit-text-fill-color:#6c63ff !important;min-width:40px;text-align:right;">{pct}%</span>
              </div>
            </div>""", unsafe_allow_html=True)

        with st.expander("📄 View Retrieved Context — Top 5 PubMed Matches"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"""
                <div style="background:#fff;border:1.5px solid #e8e8f0;border-left:4px solid #6c63ff;border-radius:14px;padding:16px 20px;margin-bottom:12px;">
                  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#6c63ff !important;-webkit-text-fill-color:#6c63ff !important;margin-bottom:8px;">Match {i} · PubMed QA</div>
                  <div style="font-size:13px;line-height:1.7;color:#555 !important;-webkit-text-fill-color:#555 !important;">{c}</div>
                </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#ccc;font-size:12px;margin-top:40px;font-family:'Nunito',sans-serif;">
  ⚠️ For educational purposes only — not a substitute for professional medical advice.<br>
  Built with FAISS · RoBERTa · Sentence Transformers · Streamlit
</div>
""", unsafe_allow_html=True)
