import streamlit as st
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="centered"
)

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main { background-color: #f0f4f8; }
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 1.5px solid #4a90d9;
            padding: 10px;
        }
        .stButton > button {
            background-color: #4a90d9;
            color: white;
            border-radius: 10px;
            padding: 8px 20px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #357abd;
        }
    </style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 Medical AI Assistant")
st.caption("Powered by PubMed QA · FAISS · Flan-T5 · Sentence Transformers")
st.divider()

# ─── Constants ─────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "faiss_index.bin"
ANSWERS_PATH     = "answers.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME   = "google/flan-t5-small"
DATASET_NAME     = "pubmed_qa"
DATASET_CONFIG   = "pqa_labeled"
DATASET_SPLIT    = "train[:500]"


# ─── Build Index (runs only if files don't exist) ──────────────────────────────
def build_and_save_index():
    """Downloads PubMed QA dataset, creates embeddings, builds FAISS index."""
    from datasets import load_dataset

    st.info("⚙️ First-time setup: Building knowledge base from PubMed QA dataset...")
    progress = st.progress(0, text="Loading dataset...")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    answers = [item['long_answer'] for item in dataset]
    progress.progress(25, text="Dataset loaded ✅  Creating embeddings...")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings  = embed_model.encode(answers, show_progress_bar=False)
    progress.progress(60, text="Embeddings done ✅  Building FAISS index...")

    dimension = embeddings.shape[1]
    index     = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    progress.progress(85, text="FAISS ready ✅  Saving to disk...")

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(ANSWERS_PATH, "wb") as f:
        pickle.dump(answers, f)

    progress.progress(100, text="Knowledge base ready! 🎉")
    st.success("Setup complete! The system is ready to answer your questions.")

    return embed_model, index, answers


# ─── Load Full System ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    try:
        # Auto-build index if not present
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(ANSWERS_PATH):
            embed_model, index, answers = build_and_save_index()
        else:
            embed_model = SentenceTransformer(EMBED_MODEL_NAME)
            index       = faiss.read_index(FAISS_INDEX_PATH)
            with open(ANSWERS_PATH, "rb") as f:
                answers = pickle.load(f)

        llm = pipeline("text2text-generation", model=LLM_MODEL_NAME)
        return embed_model, index, answers, llm

    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        return None, None, None, None


# ─── Retrieve Top-K Answers ────────────────────────────────────────────────────
def retrieve(query, embed_model, index, answers, k=3):
    query_vec = embed_model.encode([query])
    _, indices = index.search(np.array(query_vec), k)
    return [answers[i] for i in indices[0]]


# ─── Main App ──────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading AI system... (may take a moment on first run)"):
    embed_model, index, answers, llm = load_system()

if embed_model is not None:
    st.success("✅ System ready! Ask any medical question below.")
    st.write("")

    query = st.text_input(
        "💬 Enter your medical question:",
        placeholder="e.g. What are the symptoms of diabetes?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.button("🔍 Ask")

    if search_btn:
        if query.strip() == "":
            st.warning("⚠️ Please enter a question before searching.")
        else:
            with st.spinner("Thinking..."):
                retrieved = retrieve(query, embed_model, index, answers)
                context   = " ".join(retrieved)

                prompt = f"""You are a helpful medical assistant.
Answer ONLY using the context below. If the answer is not found, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""

                result = llm(prompt, max_new_tokens=200, do_sample=False)

            st.subheader("🤖 Answer")
            st.info(result[0]['generated_text'])

            with st.expander("📚 Retrieved Context (from PubMed QA)"):
                for i, r in enumerate(retrieved, 1):
                    st.markdown(f"**[{i}]** {r}")
                    st.write("")

    st.divider()
    st.caption("⚠️ This tool is for informational purposes only. Always consult a qualified medical professional.")

else:
    st.error("System failed to load. Please refresh the page or check the logs.")
