import streamlit as st
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedQuery AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CSS: ONLY decorative styles, NO text color overrides ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Nunito:wght@300;400;500;600&display=swap');

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], .stDeployButton { display:none !important; }
section[data-testid="stSidebar"] { display:none !important; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 750px !important; }

/* White background */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* Font on non-input elements only */
h1,h2,h3,h4,h5,h6,p,li,span,label,div,button {
    font-family: 'Nunito', sans-serif !important;
}

/* ── Textarea / input ── */
textarea, input[type="text"], input[type="search"] {
    background-color: #f5f5ff !important;
    border: 2px solid #e0e0f0 !important;
    border-radius: 14px !important;
    font-size: 15px !important;
    font-family: 'Nunito', sans-serif !important;
    color: #111 !important;
    caret-color: #6c63ff !important;
    padding: 14px 18px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
textarea:focus, input:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,.12) !important;
    outline: none !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #48cfad) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 12px 32px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    font-family: 'Nunito', sans-serif !important;
    box-shadow: 0 4px 18px rgba(108,99,255,.30) !important;
    transition: all .25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(108,99,255,.40) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #f8f8ff !important;
    border: 1.5px solid #e8e8f8 !important;
    border-radius: 16px !important;
    padding: 18px !important;
    text-align: center !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1.5px solid #e8e8f8 !important;
    border-radius: 14px !important;
    background: #f8f8ff !important;
}

/* ── Streamlit info box ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.8 !important;
}

/* ── Divider ── */
hr { border-color: #f0f0f8 !important; }

/* Scrollbar */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:#d0d0f0; border-radius:99px; }

/* ── Hero animations ── */
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:none; }
}
@keyframes iconPop {
    from { opacity:0; transform:scale(.5) rotate(-15deg); }
    to   { opacity:1; transform:scale(1) rotate(0); }
}
@keyframes livePulse {
    0%,100% { box-shadow:0 0 0 0 rgba(16,185,129,.5); }
    50%      { box-shadow:0 0 0 6px rgba(16,185,129,0); }
}
.hero-icon {
    width:64px; height:64px;
    background:linear-gradient(135deg,#6c63ff,#48cfad);
    border-radius:20px;
    display:inline-flex; align-items:center; justify-content:center;
    font-size:30px; margin-bottom:16px;
    box-shadow:0 8px 24px rgba(108,99,255,.25);
    animation: iconPop .7s cubic-bezier(.34,1.56,.64,1) .1s both;
}
.hero-title {
    font-family:'Playfair Display',serif !important;
    font-size:2.5rem; font-weight:600; line-height:1.2;
    background:linear-gradient(135deg,#1a1a2e,#6c63ff 60%,#48cfad);
    -webkit-background-clip:text; background-clip:text;
    color:transparent !important;
    animation: fadeUp .6s ease .2s both;
    margin-bottom:8px;
}
.hero-sub {
    font-size:15px; color:#888;
    animation: fadeUp .6s ease .3s both;
    margin-bottom:24px;
}
.status-pill {
    display:inline-flex; align-items:center; gap:7px;
    background:#f0fdf9; border:1.5px solid #bbf7e8;
    border-radius:99px; padding:5px 14px;
    font-size:11px; font-weight:700; color:#10b981;
    text-transform:uppercase; letter-spacing:.06em;
    margin-bottom:14px;
    animation: fadeUp .5s ease .1s both;
}
.live-dot {
    width:7px; height:7px; background:#10b981;
    border-radius:50%; display:inline-block;
    animation: livePulse 1.8s ease-in-out infinite;
}
.stat-box {
    background:#f8f8ff; border:1.5px solid #e8e8f8;
    border-radius:16px; padding:20px 12px; text-align:center;
    transition:transform .2s, box-shadow .2s;
    animation: fadeUp .6s ease .4s both;
}
.stat-box:hover { transform:translateY(-3px); box-shadow:0 8px 20px rgba(108,99,255,.10); }
.stat-val { font-family:'Playfair Display',serif; font-size:24px; font-weight:600; color:#6c63ff; }
.stat-lbl { font-size:11px; color:#aaa; text-transform:uppercase; letter-spacing:.07em; margin-top:4px; }
.sug-wrap {
    background:#f8f8ff; border:1.5px solid #e8e8f8;
    border-radius:16px; padding:18px 22px; margin-top:10px;
}
.sug-title { font-size:11px; font-weight:700; color:#bbb; text-transform:uppercase; letter-spacing:.1em; margin-bottom:12px; }
.sug-chips { display:flex; flex-wrap:wrap; gap:8px; }
.sug-chip {
    background:#fff; border:1.5px solid #dddcfe;
    border-radius:99px; padding:7px 16px;
    font-size:13px; font-weight:500; color:#6c63ff;
    cursor:pointer; font-family:'Nunito',sans-serif;
    transition:all .2s ease;
}
.sug-chip:hover {
    background:#6c63ff; color:#fff; border-color:#6c63ff;
    transform:translateY(-2px); box-shadow:0 4px 12px rgba(108,99,255,.25);
}
.result-label {
    display:flex; align-items:center; gap:10px; margin-bottom:12px;
}
.result-line { width:24px; height:2px; background:#6c63ff; border-radius:2px; display:inline-block; }
.result-tag {
    font-size:11px; font-weight:700; letter-spacing:.12em;
    text-transform:uppercase; color:#6c63ff;
}
.score-row {
    display:flex; align-items:center; gap:12px;
    margin:12px 0 8px; padding:14px 0 0;
    border-top:1px solid #f0f0f8;
}
.score-label { font-size:11px; font-weight:700; color:#aaa; text-transform:uppercase; letter-spacing:.08em; white-space:nowrap; }
.score-track { flex:1; height:6px; background:#ebebf8; border-radius:99px; overflow:hidden; }
.score-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#6c63ff,#48cfad); }
.score-pct { font-size:13px; font-weight:700; color:#6c63ff; min-width:36px; text-align:right; }
.ctx-label-div {
    background:#f0f0ff; border-left:4px solid #6c63ff;
    border-radius:0 10px 10px 0; padding:8px 16px; margin-bottom:6px;
}
.ctx-label-text { font-size:10px; font-weight:700; color:#6c63ff; text-transform:uppercase; letter-spacing:.1em; }
</style>

<script>
(function() {
    function fix() {
        /* Force textarea text visible */
        document.querySelectorAll('textarea,input').forEach(function(el) {
            el.style.color = '#111';
            el.style.webkitTextFillColor = '#111';
            el.style.backgroundColor = '#f5f5ff';
        });
        /* Wire suggestion chips */
        document.querySelectorAll('.sug-chip').forEach(function(chip) {
            if (chip._wired) return;
            chip._wired = true;
            chip.addEventListener('click', function() {
                var ta = document.querySelector('textarea');
                if (!ta) return;
                var setter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value').set;
                setter.call(ta, chip.innerText.trim());
                ta.dispatchEvent(new Event('input', {bubbles:true}));
                ta.focus();
            });
        });
    }
    fix();
    new MutationObserver(fix).observe(document.body, {childList:true, subtree:true});
})();
</script>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FAISS_PATH  = "faiss_index.bin"
ANS_PATH    = "answers.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:32px 0 8px;">
    <div class="status-pill"><span class="live-dot"></span> Live · PubMed QA</div>
    <div class="hero-icon">🩺</div>
    <div class="hero-title">MedQuery AI</div>
    <div class="hero-sub">Semantic search over <strong>500 PubMed</strong> research abstracts · No hallucination</div>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="stat-box"><div class="stat-val">500</div><div class="stat-lbl">PubMed Papers</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-box"><div class="stat-val">384</div><div class="stat-lbl">Vector Dims</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-box"><div class="stat-val">Top 5</div><div class="stat-lbl">Retrieved</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Load System ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    embed_model = SentenceTransformer(EMBED_MODEL)

    if not os.path.exists(FAISS_PATH) or not os.path.exists(ANS_PATH):
        bar = st.progress(0, text="⏳ Downloading PubMed QA dataset (first-time setup)…")
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train[:500]")
        answers = [item['long_answer'] for item in ds]
        bar.progress(35, text="⚙️ Generating sentence embeddings…")
        embs = embed_model.encode(answers, show_progress_bar=False)
        bar.progress(75, text="🔍 Building FAISS index…")
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(np.array(embs, dtype="float32"))
        faiss.write_index(index, FAISS_PATH)
        with open(ANS_PATH, "wb") as f:
            pickle.dump(answers, f)
        bar.progress(100, text="✅ Ready!")
    else:
        index = faiss.read_index(FAISS_PATH)
        with open(ANS_PATH, "rb") as f:
            answers = pickle.load(f)

    return embed_model, index, answers

with st.spinner("🔄 Loading AI system…"):
    try:
        embed_model, faiss_index, answers = load_system()
        system_ready = True
    except Exception as e:
        st.error(f"❌ System failed to load: {e}")
        system_ready = False

if not system_ready:
    st.stop()

# ── Retrieval ──────────────────────────────────────────────────────────────────
def search_pubmed(query: str, k: int = 5):
    vec = embed_model.encode([query])
    distances, indices = faiss_index.search(np.array(vec, dtype="float32"), k)
    passages = [answers[i] for i in indices[0]]
    # Normalise L2 distance to 0–100 relevance score
    scores = [max(0, round((1 - d / 150) * 100)) for d in distances[0]]
    return passages, scores

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("**Ask a medical question:**")
query = st.text_area(
    label="query",
    placeholder="e.g. What is diabetes?   What causes hypertension?   How does chemotherapy work?",
    height=110,
    label_visibility="collapsed"
)

btn_col, _ = st.columns([2, 3])
with btn_col:
    clicked = st.button("🔍  Search PubMed", use_container_width=True)

# ── Suggestion chips ───────────────────────────────────────────────────────────
st.markdown("""
<div class="sug-wrap">
  <div class="sug-title">Try these questions</div>
  <div class="sug-chips">
    <button class="sug-chip">What is diabetes?</button>
    <button class="sug-chip">What causes hypertension?</button>
    <button class="sug-chip">How does chemotherapy work?</button>
    <button class="sug-chip">What is insulin resistance?</button>
    <button class="sug-chip">What is a myocardial infarction?</button>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Results ────────────────────────────────────────────────────────────────────
if clicked:
    if not query.strip():
        st.warning("⚠️ Please type a medical question above.")
    else:
        with st.spinner("🔍 Searching PubMed knowledge base…"):
            passages, scores = search_pubmed(query)

        best      = passages[0]
        best_score = scores[0]

        # ── Best answer (use st.success so text is ALWAYS visible) ──
        st.markdown('<div class="result-label"><span class="result-line"></span><span class="result-tag">Most Relevant PubMed Result</span></div>', unsafe_allow_html=True)
        st.success(best)

        # ── Relevance score bar ──
        st.markdown(f"""
        <div class="score-row">
          <span class="score-label">Relevance</span>
          <div class="score-track">
            <div class="score-fill" style="width:{best_score}%"></div>
          </div>
          <span class="score-pct">{best_score}%</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── All 5 matches in expander ──
        with st.expander("📄 View all 5 PubMed matches"):
            for i, (passage, score) in enumerate(zip(passages, scores), 1):
                st.markdown(f'<div class="ctx-label-div"><span class="ctx-label-text">Match {i} &nbsp;·&nbsp; Relevance {score}%</span></div>', unsafe_allow_html=True)
                st.write(passage)
                if i < len(passages):
                    st.divider()

st.markdown("""
<div style="text-align:center; color:#ccc; font-size:12px; margin-top:40px; font-family:'Nunito',sans-serif;">
  ⚠️ For educational purposes only — not a substitute for professional medical advice.<br>
  Built with FAISS · Sentence Transformers · Streamlit &nbsp;·&nbsp; by Sonika Deshwal
</div>
""", unsafe_allow_html=True)
