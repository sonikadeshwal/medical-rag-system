# 🩺 MedQuery AI — Medical RAG System

Extractive Q&A over **500 PubMed research papers** using FAISS vector search and RoBERTa. Zero hallucination — answers are extracted directly from research text.

🔗 **Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## 🧠 How It Works

```
Your Question
      ↓
all-MiniLM-L6-v2  →  encodes query to vector
      ↓
FAISS Index  →  retrieves Top-5 similar PubMed chunks
      ↓
deepset/roberta-base-squad2  →  extracts exact answer span
      ↓
Answer + Confidence Score shown in UI
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Dataset | PubMed QA — 500 labeled samples |
| Embeddings | `all-MiniLM-L6-v2` |
| Vector Store | FAISS IndexFlatL2 |
| QA Model | `deepset/roberta-base-squad2` |
| Frontend | Streamlit + custom CSS/JS |

---

## 🚀 Run Locally

```bash
git clone https://github.com/sonikadeshwal/medical-rag-system.git
cd medical-rag-system
pip install -r requirements.txt
streamlit run app.py
```

> First run auto-builds the FAISS index (~2 min). Subsequent runs load instantly.

---

## 👩‍💻 Author

**Sonika Deshwal** · B.Tech CSE (AI & ML) · LPU

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](https://sonikadeshwal.netlify.app)
[![GitHub](https://img.shields.io/badge/GitHub-sonikadeshwal-black?logo=github)](https://github.com/sonikadeshwal)

---

⚠️ For educational purposes only. Not a substitute for professional medical advice.
