# 🩺 Medical RAG System

> A Retrieval-Augmented Generation system that answers health-related questions using trusted PubMed research data — instead of guessing.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-rag-system-yua8mtkptgcbh3sl9qu3cx.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 💡 What is a Medical RAG System?

A RAG **(Retrieval-Augmented Generation)** system combines two things:

1. **Retrieval** → finds relevant medical documents from a knowledge base
2. **Generation** → uses the retrieved documents to give a grounded answer

Instead of letting an AI guess or hallucinate, it searches real trusted sources first — then answers based on what it actually found.

---

## 🔍 How it works — Simple Flow

```
User Question
      ↓
Convert to embedding (all-MiniLM-L6-v2)
      ↓
Search in medical database (FAISS vector DB)
      ↓
Retrieve top 5 most relevant PubMed chunks
      ↓
Sentence-level re-ranking (pick best 3 sentences)
      ↓
Final grounded answer shown to user
```

---

## 🏗️ How I Built This — Step by Step

### 1. 📚 Collected Medical Data
Used the **PubMed QA dataset** from HuggingFace — 500 labeled biomedical question-answer pairs sourced from real research papers.

```python
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:500]")
answers = [item['long_answer'] for item in dataset]
```

### 2. ✂️ Chunked the Data
Each PubMed `long_answer` is used as one chunk. Why? LLMs and retrievers work better on smaller, focused pieces of text.

### 3. 🔢 Converted Text → Embeddings
Used a free HuggingFace embedding model:

```python
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings  = embed_model.encode(answers)  # 384-dimensional vectors
```

### 4. 🗄️ Stored in FAISS Vector DB
FAISS is fast and works locally — no API needed.

```python
import faiss
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
faiss.write_index(index, "faiss_index.bin")
```

### 5. 🔍 Built the Retriever
When a user asks a question:

```python
def search_pubmed(query, k=5):
    vec      = embed_model.encode([query])
    distances, indices = index.search(vec, k)
    passages = [answers[i] for i in indices[0]]
    return passages, distances
```

### 6. 🧠 Sentence-Level Re-Ranking
Instead of dumping the full passage, I split it into sentences and pick the top 3 most relevant to the actual question — using cosine similarity.

```python
# Embed each sentence → cosine similarity with query → top 3 sentences
sims   = sentence_embeddings @ query_embedding
top_3  = sorted sentences by similarity score
answer = " ".join(top_3)
```

### 7. ⚠️ Non-Medical Query Guard
If someone asks something unrelated to medicine, the FAISS distance is very high — so I block it:

```python
if best_distance > 110:
    show_warning("Please ask a medical question.")
```

---

## ⚠️ Why My Earlier Version Failed

This is what was going wrong before I fixed it:

| Problem | What happened |
|---|---|
| ❌ Bad LLM choice | `flan-t5-small` (80M params) hallucinated constantly |
| ❌ Wrong pipeline task | `text2text-generation` removed in newer `transformers` |
| ❌ No grounding | Model answered without using retrieved documents |
| ❌ Full passage shown | Dense academic text — unreadable for users |
| ❌ No relevance check | Non-medical questions still got fake medical answers |

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Dataset | [PubMed QA](https://huggingface.co/datasets/pubmed_qa) — 500 samples |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (IndexFlatL2) |
| Re-ranking | Sentence-level cosine similarity |
| Frontend | Streamlit + custom CSS/JS |
| Language | Python 3.10+ |

---

## 🚀 Run Locally

```bash
# 1. Clone
git clone https://github.com/sonikadeshwal/medical-rag-system.git
cd medical-rag-system

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

> **First run:** Automatically downloads PubMed QA, creates embeddings, builds FAISS index (~2–3 min). After that it loads instantly from cache.

---

## 📁 Project Structure

```
medical-rag-system/
│
├── app.py               # Streamlit app — full RAG pipeline + UI
├── medical_rag.ipynb    # Notebook — step-by-step walkthrough
├── requirements.txt     # Python dependencies
├── .gitignore           # Excludes faiss_index.bin, answers.pkl
└── README.md            # You're here!
```

---

## 🔮 What I'd Add Next

- 🔗 Source paper links alongside answers
- 💬 Multi-turn conversation history
- 📄 Upload your own medical PDFs as knowledge base
- 🌐 Expand to 5,000+ PubMed samples
- 🤖 Connect Claude/GPT for better generation

---

## 👩‍💻 Built by

**Sonika Deshwal**
B.Tech CSE (AI & ML) · Lovely Professional University · Batch 2023–2027

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/sonikadeshwal)
[![GitHub](https://img.shields.io/badge/GitHub-sonikadeshwal-black?logo=github)](https://github.com/sonikadeshwal)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-brightgreen?logo=netlify)](https://sonikadeshwal.netlify.app)

---

> ⚠️ **Disclaimer:** This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
