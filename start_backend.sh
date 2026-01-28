#!/bin/bash

# Tourism ChatBot Startup Script

echo "🏛️ Starting Tourism ChatBot Backend..."

# Check if virtual environment exists, if not create one
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠️  WARNING: HF_TOKEN environment variable is not set!"
    echo "Please set your Hugging Face token:"
    echo "export HF_TOKEN='your_hugging_face_token_here'"
    echo ""
    echo "You can get a token from: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Enter your Hugging Face token (or press Enter to continue without it): " token
    if [ ! -z "$token" ]; then
        export HF_TOKEN="$token"
        echo "Token set for this session."
    fi
fi

# Check if FAISS index exists
if [ ! -f "faiss_index3/index.faiss" ]; then
    echo ""
    echo "⚠️  WARNING: FAISS index not found at faiss_index3/index.faiss"
    echo "Make sure you have processed your data and created the vector store."
    echo ""
fi

echo ""
echo "Starting FastAPI server..."
echo "Backend will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload