#!/bin/bash

# Simple run script for the Tenant AI Assistant

echo "🏠 Starting Real Estate AI Tenant Assistant..."
echo "============================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix/Linux/MacOS
    source venv/bin/activate
fi

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Create documents folder if it doesn't exist
if [ ! -d "documents" ]; then
    echo "📁 Creating documents folder..."
    mkdir documents
fi

# Check if tenancy agreement exists
if [ ! -f "documents/tenancy_agreement.md" ]; then
    echo "⚠️  Warning: tenancy_agreement.md not found in documents folder"
    echo "Please add your contract document to the documents folder"
fi

# Launch the Streamlit app
echo ""
echo "🚀 Launching application..."
echo "============================================"
echo "📍 Open your browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py