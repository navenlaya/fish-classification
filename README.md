# Fish Species Classifier

A full-stack application that identifies freshwater fish species from user-uploaded images using PyTorch and transfer learning.

## Features

- **ML Model**: PyTorch 2.x with ResNet18 transfer learning
- **Backend**: FastAPI with image classification endpoint
- **Frontend**: React + TypeScript with TailwindCSS
- **Target Species**: Rainbow Trout, Largemouth Bass, Chinook Salmon

## Project Structure

```
fish-classifier/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── model.py         # Model loading and preprocessing
│   ├── requirements.txt # Python dependencies
│   └── train.py         # Model training script
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ImageUpload.tsx
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
├── README.md
└── .gitignore
```

## Setup Instructions

### 1. Model Training

First, you need to train the model with your fish image dataset:

1. **Prepare Dataset**: Create a directory structure with subfolders for each species:
   ```
   dataset/
   ├── rainbow_trout/
   ├── largemouth_bass/
   └── chinook_salmon/
   ```
   
   Each subfolder should contain at least 200 images of the respective fish species.

2. **Train Model**: Run the training script from the backend directory:
   ```bash
   cd backend
   python train.py --data_dir ../dataset --epochs 20 --batch_size 32
   ```
   
   This will create `model.pt` in the backend directory.

### 2. Backend Setup

1. **Install Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Start FastAPI Server**:
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   The API will be available at `http://localhost:8000`

### 3. Frontend Setup

1. **Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start Development Server**:
   ```bash
   cd frontend
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:5173`

## API Usage

### Classify Fish Endpoint

**POST** `/classify-fish`

**Request**: Multipart form data with image file
- `file`: Image file (JPEG, PNG, etc.)

**Response**:
```json
{
  "species": "rainbow_trout",
  "confidence": 0.95
}
```

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/classify-fish" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@fish_image.jpg"
```

## Model Details

- **Architecture**: ResNet18 with transfer learning
- **Input Size**: 224x224 pixels
- **Preprocessing**: Resize, normalize (ImageNet stats)
- **Data Augmentation**: RandomHorizontalFlip, RandomRotation, ColorJitter
- **Training**: Adam optimizer, CrossEntropyLoss

## Requirements

- Python 3.11+
- PyTorch 2.x
- CUDA-compatible GPU (optional, for faster training)
- Node.js 16+
- npm or yarn

## Troubleshooting

- **Model not found**: Ensure you've trained the model and `model.pt` exists in the backend directory
- **CUDA errors**: The model will automatically fall back to CPU if GPU is unavailable
- **Port conflicts**: Change the port numbers in the startup commands if needed
