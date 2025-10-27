# 📄 Contract Assistant

**AI-Powered Contract Analysis System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green.svg)](https://langchain.com)


> Making contract analysis simple, fast, and accurate with Large Language Models

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [API](#api-reference) • [FAQ](#faq)

---

## Overview

**Contract Assistant** is an intelligent system for contract analysis powered by LLM and RAG (Retrieval-Augmented Generation) technology. It helps you understand, analyze, and manage contract documents through:

- 💬 **Intelligent Q&A** - Ask questions in natural language
- 📝 **Auto Summarization** - Generate comprehensive or brief summaries
- 🔍 **Information Extraction** - Extract key contract details
- 📊 **Contract Comparison** - Compare multiple contracts (coming soon)

### Why Choose Us?

✅ **Accurate** - RAG-based answers with source citations  
✅ **Fast** - Sub-second responses with smart caching  
✅ **Simple** - Beautiful UI, zero learning curve  
✅ **Secure** - User data isolation, local storage  
✅ **Complete** - Full-featured contract management

---

## Features

### 📤 Smart Upload
- PDF document support
- Automatic text extraction
- Intelligent chunking
- Vector storage

### 💬 Intelligent Q&A
- Natural language queries
- Context-aware conversations
- Source citations (500 chars)
- Real-time responses

### 📝 Summarization
- **Comprehensive**: Detailed analysis
- **Brief**: Quick overview  
- **Key Points**: Structured bullets

### 🔍 Extraction
Automatically extract:
- Parties involved
- Financial terms
- Important dates
- Addresses
- Key clauses

### 👤 User System
- Secure authentication
- Data isolation
- History tracking
- File management

---

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Core language |
| Streamlit | 1.37+ | Web UI |
| LangChain | 0.2.16 | LLM framework |
| OpenAI | 1.30+ | LLM API |
| FAISS | 1.8+ | Vector DB |

---

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API Key
- 4GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/contract-assistant.git
cd contract-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
echo "OPENAI_API_KEY=your_key_here" > .env

# Run
streamlit run app.py
```

Visit: `http://localhost:8501`

### First Steps

1. **Register** an account
2. **Upload** a PDF contract  
3. **Ask** questions:
   - "What is the monthly rent?"
   - "What is the lease term?"
   - "Who are the parties?"

---

## Architecture

```
┌─────────────────────┐
│   Frontend (UI)     │
│   - Streamlit       │
└──────────┬──────────┘
           │
┌──────────┴──────────┐
│  Backend (Logic)    │
│  - Database         │
│  - File Processing  │
│  - Cache Management │
└──────────┬──────────┘
           │
┌──────────┴──────────┐
│  RAG System         │
│  - Document Loading │
│  - Vector Search    │
│  - LLM Q&A          │
└──────────┬──────────┘
           │
┌──────────┴──────────┐
│  Storage            │
│  - SQLite           │
│  - FAISS            │
│  - File System      │
└─────────────────────┘
```

### File Structure

```
contract-assistant/
├── app.py                    # Main entry
├── frontend.py               # UI layer
├── backend.py                # Business logic
├── langchain_rag_system.py  # RAG core
├── requirements.txt          # Dependencies
├── .env                      # Configuration
└── user_data/               # User files
```

---

## Documentation

### Upload Documents

1. Navigate to **Upload** tab
2. Select PDF file
3. Click **Start Processing**
4. Wait for vectorization

### Ask Questions

1. Go to **Q&A** tab
2. Type your question
3. Review answer and sources

**Tips**:
- Be specific
- One question at a time
- Verify with sources

### Generate Summaries

1. Open **Summary** tab
2. Choose type (Comprehensive/Brief/Key Points)
3. Click **Generate**

### Extract Information

1. Visit **Extract** tab
2. Click **Start Extraction**
3. View structured JSON output

---

## Configuration

### Environment Variables

`.env` file:
```bash
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
OPENAI_MODEL=gpt-4                         # Optional
```

### Model Settings

In `langchain_rag_system.py`:

```python
# Change LLM model
self.llm = ChatOpenAI(model="gpt-4", temperature=0)

# Change embedding model  
self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

### Performance Tuning

```python
chunk_size=1000      # 500-2000 recommended
chunk_overlap=200    # 100-400 recommended  
k=4                  # 2-8 documents retrieved
```

---

## API Reference

### RAG System

#### ask_question()

```python
response = rag.ask_question("What is the rent?")
# Returns: {"answer": str, "sources": List[Dict], "tokens_used": int}
```

#### summarize_contract()

```python
summary = rag.summarize_contract("comprehensive")
# Returns: str
```

#### extract_contract_info()

```python
info = rag.extract_contract_info()
# Returns: Dict with extracted fields
```

### Backend

```python
# User management
user_manager.register_user(username, email, password)
user_manager.login(username, password)

# File processing
file_processor.process_and_save_file(user_id, file, rag_system)
file_processor.load_processed_file(user_id, file_id, rag_system)

# Cache management
cache_manager.save_qa_history(user_id, file_id, question, answer, sources)
cache_manager.get_cached_summary(file_id, summary_type)
```

---

## FAQ

**Q: Installation fails?**  
A: Upgrade pip: `pip install --upgrade pip`

**Q: FAISS error?**  
A: Install CPU version: `pip install faiss-cpu`

**Q: API error?**  
A: Check API key, balance, and network

**Q: Slow responses?**  
A: Use gpt-3.5-turbo or reduce k parameter

**Q: Inaccurate answers?**  
A: Increase chunk_size or use gpt-4

---

## Roadmap

### v1.0 (Current) ✅
- [ ] RAG Q&A system
- [ ] User authentication
- [ ] Summarization
- [ ] Information extraction
- [ ] Smart caching

### v1.1 (Planned)
- [ ] Contract comparison
- [ ] Advanced search
- [ ] Export (PDF/Word/Excel)

### v1.2 (Future)
- [ ] Multi-language support
- [ ] RESTful API
- [ ] Mobile app

---

## Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [Streamlit](https://streamlit.io/) - Web framework
- [OpenAI](https://openai.com/) - LLM provider
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search

---

## Contact

- **GitHub**: [Repository](https://github.com/your-username/contract-assistant)
- **Issues**: [Issue Tracker](https://github.com/your-username/contract-assistant/issues)
- **Email**: 

---

<div align="center">

**If this project helps you, please give it a ⭐!**

Made with ❤️ 

</div>