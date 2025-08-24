from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
from model import get_model
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fish Species Classifier API",
    description="API for classifying freshwater fish species from images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model
    try:
        model = get_model()
        logger.info("Model loaded successfully")
        logger.info(f"Available classes: {model.get_class_names()}")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.warning("Please train the model first using train.py")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fish Species Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "classify": "/classify-fish",
            "health": "/health",
            "classes": "/classes"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/classes")
async def get_classes():
    """Get available fish species classes."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": model.get_class_names(),
        "count": len(model.get_class_names())
    }

@app.post("/classify-fish")
async def classify_fish(file: UploadFile = File(...)):
    """
    Classify a fish species from an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        Dictionary with predicted species and confidence
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Validate image dimensions
        if image.size[0] < 50 or image.size[1] < 50:
            raise HTTPException(
                status_code=400, 
                detail="Image too small. Minimum size is 50x50 pixels."
            )
        
        # Make prediction
        result = model.predict(image)
        
        logger.info(f"Prediction: {result['species']} with confidence {result['confidence']:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
