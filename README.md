# 🐟 Fish Species Classifier

A full-stack machine learning application that identifies fish species from user-uploaded images using PyTorch and transfer learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### One-Command Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd fish-classification

# Run the setup script (this will take several minutes)
./setup_project.sh
```

The setup script will:
- ✅ Check your system dependencies
- 📦 Create Python virtual environment
- 📦 Install all Python and Node.js dependencies
- 📥 Download the fish dataset (3.24GB)
- 🎉 Get you ready to run the application

## 🏗️ Project Structure

```
fish-classification/
├── backend/                    # Python FastAPI backend
│   ├── main.py               # FastAPI application with endpoints
│   ├── model.py              # PyTorch model loading and inference
│   ├── train.py              # Model training script
│   ├── prepare_dataset.py    # Dataset organization utility
│   └── requirements.txt      # Python dependencies
├── frontend/                  # React TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   │   └── ImageUpload.tsx  # Main image upload component
│   │   ├── App.tsx              # Main application component
│   │   └── main.tsx             # Application entry point
│   ├── package.json             # Node.js dependencies
│   └── vite.config.ts           # Vite build configuration
├── dataset/                    # Fish image dataset (created by setup)
├── setup_project.sh            # Automated setup script
├── download_kaggle_dataset.py  # Dataset download utility
├── README.md                   # This file
├── PROJECT_OVERVIEW.md         # Technical project overview
└── .gitignore                  # Git ignore patterns
```

## 🐟 Available Fish Species

The model is trained on 9 different fish species:
- **Trout** - Freshwater fish
- **Sea Bass** - Marine fish
- **Gilt-Head Bream** - Mediterranean fish
- **Red Sea Bream** - Marine fish
- **Red Mullet** - Marine fish
- **Horse Mackerel** - Marine fish
- **Black Sea Sprat** - Marine fish
- **Striped Red Mullet** - Marine fish
- **Shrimp** - Crustacean

## 🚀 Running the Application

### Start the Backend
```bash
cd backend
source ../venv/bin/activate
python main.py
```
The API will be available at `http://localhost:8000`

### Start the Frontend
```bash
cd frontend
npm run dev
```
The frontend will be available at `http://localhost:5173`

## 🔧 Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Python Backend
```bash
python3 -m venv venv
source venv/bin/activate
cd backend
pip install -r requirements.txt
```

### 2. Node.js Frontend
```bash
cd frontend
npm install
```

### 3. Download Dataset
```bash
python3 download_kaggle_dataset.py
```

## 📊 Dataset Details

- **Source**: [A Large-Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)
- **Size**: 3.24GB
- **Images**: 9,000 total (1,000 per species)
- **Resolution**: 590x445 pixels
- **Quality**: Professional research dataset with data augmentation

## 🎯 Model Details

- **Architecture**: ResNet18 with transfer learning
- **Input Size**: 224x224 pixels (resized from 590x445)
- **Preprocessing**: ImageNet normalization
- **Training**: Adam optimizer, CrossEntropyLoss
- **Data Augmentation**: Horizontal flips, rotations, color variations

## 🔌 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /classes` - Available fish species
- `POST /classify-fish` - Image classification endpoint

## 🛠️ Development

### Training the Model
```bash
cd backend
source ../venv/bin/activate
python train.py --data_dir ../dataset --epochs 20
```

### Testing
```bash
# Test backend
curl http://localhost:8000/health

# Test classification
curl -X POST "http://localhost:8000/classify-fish" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_fish_image.jpg"
```

## 📝 License

This project uses the [A Large-Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) which is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [React Documentation](https://react.dev/)
- [Dataset Paper](https://ieeexplore.ieee.org/document/9302612)

---

**🎣 Happy Fish Classification!** 🐟
