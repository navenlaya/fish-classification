import React from 'react'
import ImageUpload from './components/ImageUpload'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            🐟 Fish Species Classifier
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Upload an image of a freshwater fish and our AI will identify the species. 
            Currently supports Rainbow Trout, Largemouth Bass, and Chinook Salmon.
          </p>
        </div>
        
        <div className="max-w-2xl mx-auto">
          <ImageUpload />
        </div>
        
        <div className="mt-12 text-center text-gray-500">
          <p className="text-sm">
            Built with PyTorch, FastAPI, and React
          </p>
        </div>
      </div>
    </div>
  )
}

export default App
