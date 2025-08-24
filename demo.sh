#!/bin/bash

# Fish Species Classifier - Quick Demo Script

echo "🐟 Fish Species Classifier - Quick Demo"
echo "======================================="

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "🚀 Setting up demo environment..."

# Create demo model
echo "📦 Creating demo model..."
cd backend
python3 demo_model.py
cd ..

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install --silent
cd ..

echo ""
echo "✅ Demo setup complete!"
echo ""
echo "🌐 Starting the application..."
echo ""
echo "📋 Instructions:"
echo "1. The backend will start on http://localhost:8000"
echo "2. The frontend will start on http://localhost:5173"
echo "3. Open http://localhost:5173 in your browser"
echo "4. Upload any image to test the classification"
echo ""
echo "⚠️  Note: This demo uses a model with random weights."
echo "   Results will be random - use only for testing the UI!"
echo ""
echo "🔄 Starting services in background..."

# Start backend
cd backend
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Start frontend
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "🎉 Application started!"
echo ""
echo "📱 Frontend: http://localhost:5173"
echo "🔧 Backend: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "🛑 To stop the application, run:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "📋 Logs are saved to:"
echo "   - Backend: backend.log"
echo "   - Frontend: frontend.log"
echo ""
echo "🎣 Happy fishing! Upload some images to test the classifier!"
