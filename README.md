# 🏛️ Tourism RAG ChatBot

[![FastAPI](https://img.shields.io/badge/FastAPI-0.128.0-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.7-00D4AA?style=flat-square)](https://langchain.readthedocs.io/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Mistral](https://img.shields.io/badge/Mistral--7B-FF6B35?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTIiIGZpbGw9IndoaXRlIi8+Cjx0ZXh0IHg9IjEyIiB5PSIxNiIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjE0IiBmb250LXdlaWdodD0iYm9sZCIgZmlsbD0iIzMzMzMzMyIgdGV4dC1hbmNob3I9Im1pZGRsZSI+TTwvdGV4dD4KPC9zdmc+)](https://mistral.ai/)
[![FAISS](https://img.shields.io/badge/FAISS-1.13.2-4285F4?style=flat-square&logo=meta)](https://faiss.ai/)

An intelligent Tourism Assistant powered by **Retrieval-Augmented Generation (RAG)** technology, specifically designed to provide comprehensive information about the **Tangier-Tetouan-Hoceima** region in Morocco.

## 🌟 Features

- **🤖 Advanced AI**: Powered by Mistral-7B-Instruct-v0.2 for intelligent, contextual responses
- **🔍 RAG Architecture**: FAISS vector database with semantic search for accurate information retrieval
- **💻 Professional UI**: Modern ChatGPT-style interface with clean blue design
- **⚡ High Performance**: FastAPI backend with async processing and automatic documentation
- **🎯 Domain-Specific**: Specialized tourism knowledge for Tangier-Tetouan-Hoceima region
- **🛡️ Production Ready**: Comprehensive error handling and graceful fallbacks

## 🛠️ Technology Stack

### Core Framework
- **FastAPI** `0.128.0` - Modern, fast web framework for building APIs
- **Uvicorn** `0.40.0` - Lightning-fast ASGI server

### AI & Machine Learning
- **LangChain** `1.2.7` - Framework for developing LLM applications
- **LangChain Classic** `1.0.1` - Legacy chains compatibility
- **LangChain Community** `0.4.1` - Community integrations
- **LangChain Hugging Face** `1.2.0` - Hugging Face model integration
- **Sentence Transformers** `2.7.0` - Text embeddings generation
- **Transformers** `4.57.6` - Hugging Face transformers library
- **PyTorch** `2.10.0` - Deep learning framework

### Vector Database
- **FAISS** `1.13.2` - Efficient similarity search and clustering

### Dependencies
- **Python Multipart** `0.0.22` - Form data handling
- **Python Dotenv** - Environment variable management

## 🚀 Quick Start

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.11+** (tested with Python 3.13)
- **Hugging Face Account** - [Sign up here](https://huggingface.co/join) for model access
- **4GB+ RAM** - Required for model loading
- **Git** - For repository management

### Step 1: Clone & Navigate
```bash
git clone https://github.com/omarignammas/TourismChatbot.git
cd TourismChatbot
```

### Step 2: Environment Configuration
Create your environment file:
```bash
# Copy the environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

Add your Hugging Face token:
```env
HF_TOKEN=your_hugging_face_token_here
```

### Step 3: Automated Setup
Use our automated setup script:
```bash
# Make executable
chmod +x start_backend.sh

# Run setup and start server
./start_backend.sh
```

**What the script does:**
1. ✅ Creates isolated Python virtual environment
2. ✅ Installs all required dependencies from `requirements.txt`
3. ✅ Downloads and configures AI models (Mistral-7B, embeddings)
4. ✅ Initializes FAISS vector database
5. ✅ Starts FastAPI server on `http://localhost:8000`

### Step 4: Access Your ChatBot

| Interface | URL | Purpose |
|-----------|-----|---------|
| 🎨 **Main Application** | `frontend/professional_ui.html` | Professional ChatGPT-style interface |
| 📖 **API Docs** | `http://localhost:8000/docs` | Interactive API documentation |
| 🔗 **Health Check** | `http://localhost:8000/health` | System status monitoring |

## 🏗️ Project Architecture

```
TourismChatBot/
├── 🚀 start_backend.sh           # Automated setup & launch script
├── 📋 requirements.txt           # Python dependencies
├── ⚙️  .env                      # Environment variables (HF_TOKEN)
├── 
├── app/                          # Main application
│   ├── main.py                  # FastAPI application entry point
│   ├── models/
│   │   └── langchain_model.py   # AI models & RAG components
│   └── routers/
│       └── api.py               # Chat API endpoints
├── 
├── frontend/                     # User interfaces
│   ├── professional_ui.html     # Modern ChatGPT-style UI
│   └── simple_professional.html # Minimal clean interface
├── 
├── faiss_index/                  # Vector database storage
│   └── index.faiss              # Pre-built tourism knowledge base
├── 
├── data/                         # Data processing
│   └── process_chunk.py         # Text chunking utilities
├── 
└── 📚 README.md                 # This documentation
```

## 🔧 API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Request | Response |
|----------|--------|-------------|---------|----------|
| `/health` | GET | System health check | None | `{"status": "healthy"}` |
| `/chat/ask` | POST | Submit tourism questions | `question` form field | `{"message": "response"}` |
| `/docs` | GET | Interactive API documentation | None | Swagger UI |

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST "http://localhost:8000/chat/ask" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "question=What are the best attractions in Tangier?"
```

**Response format:**
```json
{
  "message": "Tangier offers amazing attractions including the historic Medina, Kasbah Museum, Cape Spartel lighthouse, and the beautiful beaches of the Atlantic coast..."
}
```

## 🔧 Manual Setup (Alternative)

If you prefer manual setup or troubleshooting:

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Setup
```bash
export HF_TOKEN="your_hugging_face_token_here"
# Or add to .env file
echo "HF_TOKEN=your_token" > .env
```

### 4. Start Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🧪 Development & Testing

### Development Mode
```bash
# Start with auto-reload for development
uvicorn app.main:app --reload --log-level debug

# Check model loading
python -c "from app.models.langchain_model import *; print('Models loaded successfully!')"
```

### Testing Endpoints
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test chat functionality
curl -X POST http://localhost:8000/chat/ask \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "question=Tell me about Chefchaouen"
```

### Model Information
| Component | Model/Version | Purpose |
|-----------|---------------|---------|
| **LLM** | `mistralai/Mistral-7B-Instruct-v0.2` | Text generation & conversation |
| **Embeddings** | `sentence-transformers` model | Text vectorization for search |
| **Vector DB** | FAISS with similarity search | Knowledge retrieval |
| **Chain Type** | RetrievalQA with stuff documents | RAG implementation |

## 🚨 Troubleshooting

### Common Issues & Solutions

#### Model Loading Errors
```bash
# Issue: HuggingFace authentication error
# Solution: Verify your HF_TOKEN
python -c "import os; print('HF_TOKEN:', os.getenv('HF_TOKEN', 'NOT_SET'))"

# Issue: CUDA/GPU memory errors  
# Solution: The app automatically falls back to CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

#### Dependency Conflicts
```bash
# Issue: Package version conflicts
# Solution: Create fresh environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Port Already in Use
```bash
# Issue: Port 8000 occupied
# Solution: Use different port
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

#### Model Download Issues
```bash
# Issue: Slow or failed model downloads
# Solution: Pre-download models
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

print('Downloading LLM...')
AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')

print('Downloading embeddings model...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('✅ All models downloaded!')
"
```

### Performance Optimization

#### Memory Usage
- **Minimum**: 4GB RAM (CPU-only mode)
- **Recommended**: 8GB+ RAM for optimal performance
- **GPU**: Optional, automatically detected and used

#### Response Time
- **First request**: 30-60 seconds (model loading)
- **Subsequent requests**: 2-10 seconds (depending on complexity)
- **Optimization**: Keep server running for production use

## 🔄 Stopping the Application

```bash
# Graceful shutdown (in terminal with server)
Ctrl + C

# Force stop (from another terminal)
pkill -f uvicorn

# Stop specific port
lsof -ti:8000 | xargs kill -9
```

## 🛠️ Customization

### Adding New Tourism Data
1. Prepare text files with tourism information
2. Update `data/process_chunk.py` with new data
3. Rebuild FAISS index:
```bash
python data/process_chunk.py
```

### Modifying UI
- **Professional Interface**: Edit `frontend/professional_ui.html`
- **Colors & Styling**: Modify CSS variables in the HTML files
- **Chat Behavior**: Update JavaScript functions

### API Customization
- **Response Format**: Modify `app/routers/api.py`
- **New Endpoints**: Add routes in the routers directory
- **Model Settings**: Adjust parameters in `app/models/langchain_model.py`

## 📞 Support & Community

### Getting Help
1. **Documentation**: Check this README and `/docs` endpoint
2. **Issues**: Open issues on [GitHub](https://github.com/omarignammas/TourismChatbot/issues)
3. **API Docs**: Visit `http://localhost:8000/docs` when server is running

### Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mistral AI** - For the powerful Mistral-7B language model
- **LangChain** - For the RAG framework and tools
- **Hugging Face** - For model hosting and transformers library
- **FastAPI** - For the modern web framework
- **FAISS** - For efficient vector similarity search

---

**🚀 Ready to explore Morocco's tourism with AI? Start your chatbot now!**