# 🩺 Medical AI Assistant — RAG-based Q&A System

A Retrieval-Augmented Generation (RAG) system that answers medical questions using the **PubMed QA** dataset, **FAISS** vector search, and **Flan-T5** as the language model — all wrapped in a clean **Streamlit** web app.

> ⚠️ This tool is for informational and educational purposes only. Always consult a qualified medical professional.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 🧠 How It Works

```
User Query
    │
    ▼
Sentence Transformer (all-MiniLM-L6-v2)
    │  encodes query into a vector
    ▼
FAISS Vector Search
    │  retrieves top-3 most similar answers from PubMed QA
    ▼
Flan-T5 (LLM)
    │  generates a grounded answer using retrieved context
    ▼
Streamlit UI displays the answer + retrieved context
```

This is a classic **RAG pipeline**:
- **Retrieval** → FAISS finds relevant medical passages
- **Augmentation** → passages are injected as context into the prompt
- **Generation** → Flan-T5 generates a factual answer grounded in evidence

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Dataset | [PubMed QA](https://huggingface.co/datasets/pubmed_qa) (500 labeled samples) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Language Model | `google/flan-t5-small` |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
medical-rag/
│
├── app.py                  # Streamlit web app (auto-builds index on first run)
├── medical_rag.ipynb       # Notebook: data pipeline, embedding, FAISS, LLM
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes large binary files
└── README.md               # You are here
```

---

## ⚡ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/sonikadeshwal/medical-rag.git
cd medical-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

> **Note:** On first run, the app will automatically download the PubMed QA dataset, generate embeddings, and build the FAISS index. This takes ~2–3 minutes. Subsequent runs load instantly from cache.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → set `app.py` as the entry point
5. Click **Deploy** ✅

No manual setup needed — the app auto-builds the knowledge base on first boot.

---

## 📊 Dataset

**PubMed QA** (`pqa_labeled`) from HuggingFace:
- Biomedical question-answer pairs sourced from PubMed research papers
- Uses `long_answer` fields as the retrieval corpus
- 500 samples used for this demo (extendable)

---

## 🔮 Future Improvements

- [ ] Swap Flan-T5 for a stronger open-source LLM (e.g., Mistral, LLaMA)
- [ ] Add source paper links alongside retrieved context
- [ ] Expand dataset beyond 500 samples
- [ ] Add conversation history / multi-turn Q&A
- [ ] Add confidence scores for retrieved answers

---

## 👩‍💻 Author

**Sonika Deshwal**
B.Tech CSE (AI & ML) — Lovely Professional University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-sonikadeshwal-black?logo=github)](https://github.com/sonikadeshwal)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](https://sonikadeshwal.netlify.app)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
