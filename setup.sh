#!/bin/bash

# Insurance FAQ Chatbot - Complete Setup Script
# This script sets up the entire system from scratch

set -e  # Exit on error

echo "=================================="
echo "Insurance FAQ Chatbot Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# # Check Python version
# echo "Step 1: Checking Python version..."
# if command -v python3 &> /dev/null; then
#     PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
#     print_success "Python 3 found (version $PYTHON_VERSION)"
# else
#     print_error "Python 3 not found. Please install Python 3.10 or higher."
#     exit 1
# fi

# # Create virtual environment
# echo ""
# echo "Step 2: Creating virtual environment..."
# if [ ! -d "venv" ]; then
#     python3 -m venv venv
#     print_success "Virtual environment created"
# else
#     print_info "Virtual environment already exists"
# fi

# # Activate virtual environment
# echo ""
# echo "Step 3: Activating virtual environment..."
# source venv/bin/activate
# print_success "Virtual environment activated"

# # Upgrade pip
# echo ""
# echo "Step 4: Upgrading pip..."
# pip install --upgrade pip > /dev/null 2>&1
# print_success "pip upgraded"

# # Install dependencies
# echo ""
# echo "Step 5: Installing dependencies..."
# print_info "This may take a few minutes..."
# pip install -r requirements.txt > /dev/null 2>&1
# print_success "Dependencies installed"

# # Create .env file
# echo ""
# echo "Step 6: Setting up environment..."
# if [ ! -f ".env" ]; then
#     cp .env.example .env
#     print_success ".env file created"
#     print_info "Please edit .env and add your ANTHROPIC_API_KEY"
# else
#     print_info ".env file already exists"
# fi

# # Create directories
# echo ""
# echo "Step 7: Creating directories..."
# python3 src/scripts/init_db.py
# print_success "Directories created"

# Download embedding model
echo ""
echo "Step 8: Downloading embedding model..."
print_info "First run will download the model (one-time, ~100MB)"
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" > /dev/null 2>&1
print_success "Embedding model ready"

# Final instructions
echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your API key:"
echo "   Edit .env and set ANTHROPIC_API_KEY=your_key_here"
echo ""
echo "2. Add PDF documents:"
echo "   cp your_documents.pdf data/pdfs/"
echo ""
echo "3. Ingest documents:"
echo "   python src/scripts/ingest_documents.py"
echo ""
echo "4. Start the chatbot:"
echo "   streamlit run src/app.py"
echo ""
echo "5. Run tests:"
echo "   pytest tests/"
echo ""
echo "=================================="
echo ""
print_success "Happy chatting! 🎉"