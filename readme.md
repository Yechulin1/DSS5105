# Smart Contract Assistant (RAG)

An AI-powered rental contract assistant built on Retrieval-Augmented Generation (RAG). It supports multi-user login, PDF contract upload, semantic search, Q&A, summaries, and key information extraction with local caching and SQLite persistence.

---

## 🚀 Quick Start (Usage First)

### 1) Environment
- Python 3.8+
- Optional: OpenAI API key (for GPT models)

### 2) Setup

PowerShell (Windows):

```powershell
git clone https://github.com/Yechulin1/DSS5105/tree/yechulin
cd DSS5105
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
Copy-Item env_example.txt .env
# edit .env and set at least:
# OPENAI_API_KEY=your_api_key
# OPENAI_MODEL=gpt-3.5-turbo
```

macOS/Linux:

```bash
git clone https://github.com/Yechulin1/DSS5105/tree/yechulin
cd DSS5105
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env_example.txt .env
# edit .env and set at least OPENAI_API_KEY, OPENAI_MODEL
```

### 3) Run

```powershell
streamlit run app.py
```

Open http://localhost:8501

### 4) How to Use (in app)
1. Register → Login
2. Upload a PDF contract (Upload tab)
3. Ask questions about the contract (Q&A tab)
4. Generate summaries (Summary tab): brief/comprehensive/key points
5. Extract key fields like rent, deposit, dates (Extract tab)

---

## ✨ Features
- Multi-user system: register/login, per-user data isolation
- PDF ingestion: parsing, chunking, embeddings, FAISS vector store
- Contract Q&A with source citations and short-term conversation memory
- Summaries: brief, comprehensive, key points (cached by file+type)
- Key information extraction (rent, deposit, duration, fees, policies)
- Caching across layers: in-memory, FAISS on disk, SQLite tables
- Secure password hashing (PBKDF2-HMAC-SHA256 with salt)

---

## 🏗️ Architecture

Technology stack:
- UI: Streamlit (`app.py`)
- Business logic: `backend.py` (users, files, cache)
- AI/RAG: `langchain_rag_system.py` (PDF load, embeddings, retrieval, LLM)
- Data: SQLite (structured data), FAISS (vector search), filesystem (documents)

Project layout (key parts):

```
5105_v10_zyh/
├─ app.py                    # Streamlit UI entry
├─ backend.py                # DB/users/files/cache orchestration
├─ langchain_rag_system.py   # AdvancedContractRAG (RAG engine)
├─ requirements.txt
├─ env_example.txt  → .env   # environment settings
├─ user_data/
│  └─ {user_id}/
│     ├─ contracts/
│     ├─ vector_stores/
│     └─ cache/
└─ contract_system.db        # auto-created SQLite database
```

Data model (tables): `users`, `processed_files`, `cached_summaries`, `qa_history`, `extracted_info_cache`.

---

## 🔑 Key Files
- `app.py`: Streamlit app (tabs: Upload, Q&A, Summary, Extract, History)
- `backend.py`:
   - DatabaseManager: create/migrate tables
   - UserManager: register/login, role (tenant/landlord)
   - FileProcessor: upload → parse → embed → persist → load
   - CacheManager: summaries/extractions/Q&A history
- `langchain_rag_system.py`:
   - `AdvancedContractRAG`: `load_pdf`, `ask_question`, `summarize_contract`, `extract_key_information`, `clear_all_documents`, vector store save/load

---

## ⚙️ Notes & Tips
- First run may take longer to embed; subsequent loads use FAISS/vector cache.
- Each new upload clears prior in-memory docs to avoid context mixing.
- If OpenAI is not configured, ensure `OPENAI_API_KEY` and `OPENAI_MODEL` exist in `.env`.
- User data is isolated under `user_data/{user_id}`; do not commit this folder.

---

## 🔮 Roadmap (Short & Sweet)
- Multi-document compare and cross-document search
- OCR for scanned PDFs and better table extraction
- API endpoints (upload/ask/summary/extract)
- Role-tailored prompts and question templates (tenant/landlord)
- Export reports (PDF/Word) and shareable links

---

## 📝 License

MIT License

