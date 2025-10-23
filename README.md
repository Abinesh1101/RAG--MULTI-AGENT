# 🤖 Multi-Agent RAG System

> Vision-Enhanced, Intent-Aware AI Assistant for Intelligent Document Q&A

An enterprise-grade multi-agent Retrieval-Augmented Generation (RAG) system that combines **ML-based intent classification**, **multimodal document retrieval**, **BLIP-2 vision-language understanding**, and **Mistral 7B reasoning**.  
It intelligently answers natural-language queries from PDFs and images — handling fact-based, analytical, summary, and visual insights.

---

## ✨ Features

- 🎯 **Intent Classification:** Logistic Regression + TF-IDF (≈90% accuracy)
- 📚 **Multimodal Retrieval:** FAISS-based semantic search (text + images)
- 👁️ **Vision Understanding:** BLIP-2 for interpreting charts and diagrams
- 🧠 **Reasoning Engine:** Context synthesis using Mistral 7B (Ollama)
- ⚙️ **Agent Orchestration:** Five specialized AI agents coordinated via controller logic
- 💻 **Interactive UI:** Built with Streamlit for live queries and analysis

---

## 🧱 System Architecture

```
User Query
    ↓
Intent Agent (Classification)
    ↓
Controller Agent (Routing)
    ↓
 ┌───────────────────────────────────────────────┐
 │ Retrieval Agent │ Vision Agent │ Reasoning Agent │
 └───────────────────────────────────────────────┘
    ↓
Final Answer (with sources)
```

---

## 📁 Project Structure

```
RAG-MULTI-AGENT/
├── agents/
│   ├── intent_agent.py
│   ├── retrieval_agent.py
│   ├── vision_agent.py
│   ├── reasoning_agent.py
│   └── controller_agent.py
│
├── data/
│   ├── pdfs/
│   ├── images/
│   └── faiss_store.npz
│
├── models/
│   ├── intent_model.pkl
│   ├── vectorizer.pkl
│   └── train_intent_model1.py
│
├── app.py
├── main.py
├── extracted_images.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/RAG-MULTI-AGENT.git
cd RAG-MULTI-AGENT
```

### 2️⃣ Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Ollama (for Mistral 7B)
```bash
ollama serve
ollama pull mistral
```

### 5️⃣ Run Streamlit UI
```bash
streamlit run app.py
```

---

## 🚀 Usage

### Example Queries
| Type | Example Query |
|------|----------------|
| 🧾 Fact | "How many vehicles did Tesla deliver in 2023?" |
| 📈 Analysis | "Compare Amazon’s carbon intensity between 2022 and 2023." |
| 📰 Summary | "Summarize key sustainability achievements in 2023." |
| 📊 Visual | "What does the emissions chart show?" |

---

## 🧩 How It Works

| Stage | Description |
|-------|--------------|
| **Intent Detection** | Classifies query type into fact, analysis, summary, or visual |
| **Retrieval** | Uses FAISS + embeddings to fetch relevant text/image data |
| **Vision Understanding** | BLIP-2 interprets images, graphs, or infographics |
| **Reasoning** | Mistral 7B LLM generates final answer |
| **Controller** | Manages workflow between agents |

---

## 📊 Performance

| Metric | Value | Notes |
|--------|--------|-------|
| Intent Accuracy | ~90% | 4-class classification |
| Retrieval Precision@5 | ~85% | Top 5 results relevance |
| Text Chunks | 2.2K | From 3 PDFs |
| Images | 140 | Extracted with PyMuPDF |
| Query Time (CPU) | 2–5 min | End-to-end |
| Query Time (GPU) | 10–30 sec | With CUDA |

---


## 🌐 Technologies Used

- **Python 3.11**
- **Scikit-learn** – Intent classification
- **SentenceTransformers** – Embeddings
- **FAISS** – Vector similarity search
- **BLIP-2 (Salesforce)** – Vision-language model
- **Ollama Mistral 7B** – Reasoning LLM
- **Streamlit** – Web interface
- **PyMuPDF** – PDF processing

---

## 🧪 Future Enhancements

- [ ] Conversational memory for multi-turn Q&A  
- [ ] FastAPI integration for REST API  
- [ ] Cross-modal attention reranking  
- [ ] Dashboard for performance visualization  
- [ ] Support for Word, Excel, and PowerPoint files

---

## 📧 Contact

**Author:** Abinesh Sankaranarayanan  
**Email:** abiunni0209@gmail.com 
**GitHub:** https://github.com/Abinesh1101  

⭐ If you like this project, consider giving it a star!

---

© 2025 Multi-Agent RAG Project — All rights reserved.
