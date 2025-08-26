#!/bin/bash

# Fish Classification Project Setup Script
# This script sets up the project environment and downloads the dataset

echo "🐟 Fish Classification Project Setup"
echo "===================================="

# Check if Python and Node.js are installed
echo "✅ Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Python $(python3 --version | cut -d' ' -f2)"
echo "✅ Node.js $(node --version)"
echo "✅ npm $(npm --version)"

# Create virtual environment
echo ""
echo "📦 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Download dataset
echo ""
echo "📥 Downloading fish dataset..."
echo "⏳ This may take several minutes depending on your internet speed..."
python3 download_kaggle_dataset.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 To start the project:"
echo "  1. Backend: cd backend && source ../venv/bin/activate && python main.py"
echo "  2. Frontend: cd frontend && npm run dev"
echo ""
echo "📚 For more information, see README.md"
