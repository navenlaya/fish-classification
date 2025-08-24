import React, { useState, useRef } from 'react'

interface ClassificationResult {
  species: string
  confidence: number
}

const ImageUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<ClassificationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file')
        return
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB')
        return
      }

      setSelectedFile(file)
      setError(null)
      setResult(null)

      // Create preview URL
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    }
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file) {
      setSelectedFile(file)
      setError(null)
      setResult(null)

      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    }
  }

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
  }

  const handleSubmit = async () => {
    if (!selectedFile) return

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch('http://localhost:8000/classify-fish', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to classify image')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const getSpeciesDisplayName = (species: string) => {
    return species
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-success-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-error-600'
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* File Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          selectedFile
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {!selectedFile ? (
          <div>
            <div className="text-6xl mb-4">📁</div>
            <p className="text-lg text-gray-600 mb-2">
              Drag and drop an image here, or click to browse
            </p>
            <p className="text-sm text-gray-500 mb-4">
              Supports JPEG, PNG, GIF (max 10MB)
            </p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 transition-colors"
            >
              Choose Image
            </button>
          </div>
        ) : (
          <div>
            <div className="text-6xl mb-4">✅</div>
            <p className="text-lg text-gray-600 mb-2">
              {selectedFile.name}
            </p>
            <p className="text-sm text-gray-500">
              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        )}
        
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />
      </div>

      {/* Preview */}
      {previewUrl && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Image Preview</h3>
          <div className="relative">
            <img
              src={previewUrl}
              alt="Preview"
              className="max-w-full h-auto max-h-64 rounded-lg shadow-md mx-auto"
            />
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-4 bg-error-50 border border-error-200 rounded-lg">
          <p className="text-error-600">{error}</p>
        </div>
      )}

      {/* Action Buttons */}
      {selectedFile && (
        <div className="mt-6 flex gap-3 justify-center">
          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <span className="flex items-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Classifying...
              </span>
            ) : (
              'Classify Fish'
            )}
          </button>
          <button
            onClick={handleClear}
            className="bg-gray-500 text-white px-6 py-2 rounded-lg hover:bg-gray-600 transition-colors"
          >
            Clear
          </button>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="mt-6 p-6 bg-success-50 border border-success-200 rounded-lg">
          <h3 className="text-xl font-semibold text-success-800 mb-4">
            Classification Result
          </h3>
          <div className="text-center">
            <div className="text-4xl mb-2">🐟</div>
            <p className="text-2xl font-bold text-success-700 mb-2">
              {getSpeciesDisplayName(result.species)}
            </p>
            <p className={`text-lg font-semibold ${getConfidenceColor(result.confidence)}`}>
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default ImageUpload
