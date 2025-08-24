#!/usr/bin/env python3
"""
Test Script for Fish Species Classifier

This script tests various components of the application to ensure
everything is working correctly.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_backend_dependencies():
    """Test if backend dependencies can be imported."""
    print("🔍 Testing backend dependencies...")
    
    try:
        import torch
        import torchvision
        import fastapi
        import uvicorn
        import PIL
        print("✅ All backend dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Backend dependency import failed: {e}")
        return False

def test_model_creation():
    """Test if we can create a demo model."""
    print("\n🔍 Testing model creation...")
    
    try:
        # Change to backend directory
        os.chdir("backend")
        
        # Create demo model
        subprocess.run([sys.executable, "demo_model.py"], check=True, capture_output=True)
        
        # Check if model file exists
        if os.path.exists("model.pt"):
            print("✅ Demo model created successfully")
            os.chdir("..")
            return True
        else:
            print("❌ Demo model file not found")
            os.chdir("..")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Model creation failed: {e}")
        os.chdir("..")
        return False

def test_backend_startup():
    """Test if the backend can start up."""
    print("\n🔍 Testing backend startup...")
    
    try:
        # Change to backend directory
        os.chdir("backend")
        
        # Start backend in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "127.0.0.1", "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(5)
        
        # Test if backend is responding
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=10)
            if response.status_code == 200:
                print("✅ Backend started and responding")
                process.terminate()
                process.wait()
                os.chdir("..")
                return True
            else:
                print(f"❌ Backend responded with status: {response.status_code}")
                process.terminate()
                process.wait()
                os.chdir("..")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Backend not responding: {e}")
            process.terminate()
            process.wait()
            os.chdir("..")
            return False
            
    except Exception as e:
        print(f"❌ Backend startup test failed: {e}")
        os.chdir("..")
        return False

def test_frontend_dependencies():
    """Test if frontend dependencies can be installed."""
    print("\n🔍 Testing frontend dependencies...")
    
    try:
        # Change to frontend directory
        os.chdir("frontend")
        
        # Check if node_modules exists, if not install dependencies
        if not os.path.exists("node_modules"):
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True, capture_output=True)
        
        print("✅ Frontend dependencies ready")
        os.chdir("..")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend dependency installation failed: {e}")
        os.chdir("..")
        return False
    except Exception as e:
        print(f"❌ Frontend dependency test failed: {e}")
        os.chdir("..")
        return False

def test_api_endpoints():
    """Test the API endpoints."""
    print("\n🔍 Testing API endpoints...")
    
    try:
        # Test health endpoint
        response = requests.get("http://127.0.0.1:8000/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test classes endpoint
        response = requests.get("http://127.0.0.1:8000/classes")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Classes endpoint working: {data['classes']}")
        else:
            print(f"❌ Classes endpoint failed: {response.status_code}")
            return False
        
        # Test root endpoint
        response = requests.get("http://127.0.0.1:8000/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API endpoint test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🐟 Fish Species Classifier - Application Test")
    print("=" * 50)
    
    tests = [
        ("Backend Dependencies", test_backend_dependencies),
        ("Model Creation", test_model_creation),
        ("Backend Startup", test_backend_startup),
        ("Frontend Dependencies", test_frontend_dependencies),
        ("API Endpoints", test_api_endpoints),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
        print("\n🚀 To start the application:")
        print("1. Backend: cd backend && uvicorn main:app --reload --port 8000")
        print("2. Frontend: cd frontend && npm run dev")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("💡 Make sure you have all dependencies installed and the backend is running.")

if __name__ == "__main__":
    main()
