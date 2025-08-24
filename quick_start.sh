#!/bin/bash

# Fish Species Classifier - Quick Start Script

echo "🐟 Fish Species Classifier - Quick Start"
echo "========================================"

# Check if Python 3.11+ is installed
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
if [[ $(echo "$python_version >= 3.11" | bc -l) -eq 0 ]]; then
    echo "❌ Error: Python 3.11 or higher is required. Current version: $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi
echo "✅ Node.js is installed"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed. Please install npm first."
    exit 1
fi
echo "✅ npm is installed"

echo ""
echo "🚀 Setting up the application..."

# Create virtual environment for Python
echo "📦 Creating Python virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create dataset directory structure
echo "📁 Creating dataset directory structure..."
python prepare_dataset.py --action create

echo ""
echo "✅ Backend setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add your fish images to the dataset directory:"
echo "   - Place rainbow trout images in: ../dataset/train/rainbow_trout/ and ../dataset/val/rainbow_trout/"
echo "   - Place largemouth bass images in: ../dataset/train/largemouth_bass/ and ../dataset/val/largemouth_bass/"
echo "   - Place chinook salmon images in: ../dataset/train/chinook_salmon/ and ../dataset/val/chinook_salmon/"
echo ""
echo "2. Train the model:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python train.py --data_dir ../dataset --epochs 20"
echo ""
echo "3. Start the backend server:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "4. In a new terminal, start the frontend:"
echo "   cd frontend"
echo "   npm install"
echo "   npm run dev"
echo ""
echo "🌐 The application will be available at:"
echo "   - Frontend: http://localhost:5173"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "🎯 Happy fishing! 🎣"
