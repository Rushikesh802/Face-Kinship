import React, { useState, useRef } from 'react';
import { Upload, Scan, Users, Sparkles, AlertCircle, CheckCircle2, Loader2, X } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:5000';

function App() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [preview1, setPreview1] = useState(null);
  const [preview2, setPreview2] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver1, setDragOver1] = useState(false);
  const [dragOver2, setDragOver2] = useState(false);
  const [validating1, setValidating1] = useState(false);
  const [validating2, setValidating2] = useState(false);
  const [validation1, setValidation1] = useState(null);
  const [validation2, setValidation2] = useState(null);

  const fileInput1Ref = useRef(null);
  const fileInput2Ref = useRef(null);

  const validateFaceInImage = async (file, imageNumber) => {
    const setValidating = imageNumber === 1 ? setValidating1 : setValidating2;
    const setValidation = imageNumber === 1 ? setValidation1 : setValidation2;

    setValidating(true);
    setValidation(null);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await axios.post(`${API_URL}/validate`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.valid) {
        setValidation({ valid: true, message: 'Face detected ✓' });
      } else {
        setValidation({ valid: false, message: response.data.error });
      }
    } catch (err) {
      console.error('Validation error:', err);
      setValidation({ 
        valid: false, 
        message: err.response?.data?.error || 'Validation failed. Please try again.' 
      });
    } finally {
      setValidating(false);
    }
  };

  const handleFileSelect = async (file, imageNumber) => {
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file');
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setError('Image size should be less than 5MB');
      return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
      if (imageNumber === 1) {
        setImage1(file);
        setPreview1(reader.result);
      } else {
        setImage2(file);
        setPreview2(reader.result);
      }
      setError(null);
      setResult(null);
    };
    reader.readAsDataURL(file);

    // Validate face immediately
    await validateFaceInImage(file, imageNumber);
  };

  const handleDragOver = (e, imageNumber) => {
    e.preventDefault();
    if (imageNumber === 1) {
      setDragOver1(true);
    } else {
      setDragOver2(true);
    }
  };

  const handleDragLeave = (e, imageNumber) => {
    e.preventDefault();
    if (imageNumber === 1) {
      setDragOver1(false);
    } else {
      setDragOver2(false);
    }
  };

  const handleDrop = (e, imageNumber) => {
    e.preventDefault();
    if (imageNumber === 1) {
      setDragOver1(false);
    } else {
      setDragOver2(false);
    }

    const file = e.dataTransfer.files[0];
    handleFileSelect(file, imageNumber);
  };

  const handleRemoveImage = (imageNumber) => {
    if (imageNumber === 1) {
      setImage1(null);
      setPreview1(null);
      setValidation1(null);
      setValidating1(false);
      if (fileInput1Ref.current) fileInput1Ref.current.value = '';
    } else {
      setImage2(null);
      setPreview2(null);
      setValidation2(null);
      setValidating2(false);
      if (fileInput2Ref.current) fileInput2Ref.current.value = '';
    }
    setResult(null);
  };

  const handleAnalyze = async () => {
    if (!image1 || !image2) {
      setError('Please upload both images');
      return;
    }

    // Check if both images passed validation
    if (!validation1?.valid || !validation2?.valid) {
      setError('Please upload valid face images before analyzing');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image1', image1);
      formData.append('image2', image2);

      const response = await axios.post(`${API_URL}/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      console.error('Error analyzing images:', err);
      setError(
        err.response?.data?.error || 
        'Failed to analyze images. Please make sure the backend server is running.'
      );
    } finally {
      setLoading(false);
    }
  };

  const ImageUploadBox = ({ imageNumber, preview, dragOver }) => {
    const validating = imageNumber === 1 ? validating1 : validating2;
    const validation = imageNumber === 1 ? validation1 : validation2;

    return (
      <div
        className={`relative group ${
          dragOver ? 'scale-105' : ''
        } transition-all duration-300`}
        onDragOver={(e) => handleDragOver(e, imageNumber)}
        onDragLeave={(e) => handleDragLeave(e, imageNumber)}
        onDrop={(e) => handleDrop(e, imageNumber)}
      >
        <div
          className={`relative h-80 rounded-2xl border-2 border-dashed overflow-hidden
            ${dragOver 
              ? 'border-cyber-blue bg-cyber-blue/10' 
              : validation?.valid 
                ? 'border-green-500/60 hover:border-green-500'
                : validation?.valid === false
                  ? 'border-red-500/60 hover:border-red-500'
                  : 'border-cyber-blue/30 hover:border-cyber-blue/60'
            }
            ${preview ? 'border-solid' : ''}
            transition-all duration-300 cursor-pointer glass glow-border`}
          onClick={() => {
            if (!preview) {
              imageNumber === 1 ? fileInput1Ref.current?.click() : fileInput2Ref.current?.click();
            }
          }}
        >
          {preview ? (
            <>
              <img
                src={preview}
                alt={`Face ${imageNumber}`}
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemoveImage(imageNumber);
                  }}
                  className="absolute top-4 right-4 p-2 bg-red-500/80 hover:bg-red-600 rounded-full transition-colors"
                >
                  <X size={20} />
                </button>
              </div>
              <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/90 to-transparent">
                {validating ? (
                  <div className="flex items-center gap-2">
                    <Loader2 size={16} className="animate-spin text-cyber-blue" />
                    <p className="text-sm text-cyber-blue font-semibold">Validating face...</p>
                  </div>
                ) : validation?.valid ? (
                  <div className="flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-green-400" />
                    <p className="text-sm text-green-400 font-semibold">{validation.message}</p>
                  </div>
                ) : validation?.valid === false ? (
                  <div className="flex items-center gap-2">
                    <AlertCircle size={16} className="text-red-400" />
                    <p className="text-sm text-red-400 font-semibold">{validation.message}</p>
                  </div>
                ) : (
                  <p className="text-sm text-cyber-blue font-semibold">Face {imageNumber} Uploaded</p>
                )}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full p-8 text-center">
              <div className="mb-4 p-6 rounded-full bg-cyber-blue/10 group-hover:bg-cyber-blue/20 transition-colors">
                <Upload size={48} className="text-cyber-blue animate-float" />
              </div>
              <h3 className="text-xl font-bold mb-2 text-white">
                Upload Face {imageNumber}
              </h3>
              <p className="text-gray-400 mb-4">
                Drag & drop or click to browse
              </p>
              <p className="text-xs text-gray-500">
                Supports: JPG, PNG, JPEG (Max 5MB)
              </p>
            </div>
          )}
        </div>
        
        <input
          ref={imageNumber === 1 ? fileInput1Ref : fileInput2Ref}
          type="file"
          accept="image/*"
          onChange={(e) => handleFileSelect(e.target.files[0], imageNumber)}
          className="hidden"
        />
      </div>
    );
  };

  return (
    <div className="min-h-screen animated-gradient">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyber-blue/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyber-purple/5 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Scan size={48} className="text-cyber-blue mr-4 animate-pulse" />
            <h1 className="text-5xl md:text-6xl font-bold gradient-text">
              Face Kinship Verification
            </h1>
          </div>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Advanced AI-powered facial analysis to determine biological relationships using deep learning
          </p>
          <div className="flex items-center justify-center gap-4 mt-6">
            <div className="flex items-center gap-2 px-4 py-2 rounded-full glass">
              <Sparkles size={16} className="text-cyber-blue" />
              <span className="text-sm text-gray-300">Siamese CNN</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full glass">
              <Users size={16} className="text-cyber-purple" />
              <span className="text-sm text-gray-300">KinFaceW-II Dataset</span>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="max-w-6xl mx-auto">
          {/* Upload Section */}
          <div className="grid md:grid-cols-2 gap-8 mb-8">
            <ImageUploadBox imageNumber={1} preview={preview1} dragOver={dragOver1} />
            <ImageUploadBox imageNumber={2} preview={preview2} dragOver={dragOver2} />
          </div>

          {/* Analyze Button */}
          <div className="text-center mb-8">
            <button
              onClick={handleAnalyze}
              disabled={!image1 || !image2 || loading || !validation1?.valid || !validation2?.valid || validating1 || validating2}
              className={`
                px-12 py-4 rounded-full text-lg font-bold
                bg-gradient-to-r from-cyber-blue via-cyber-purple to-cyber-pink
                hover:shadow-2xl hover:shadow-cyber-blue/50
                transform hover:scale-105 active:scale-95
                transition-all duration-300
                disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
                relative overflow-hidden group
              `}
            >
              <span className="relative z-10 flex items-center gap-3">
                {loading ? (
                  <>
                    <Loader2 size={24} className="animate-spin" />
                    Analyzing...
                  </>
                ) : validating1 || validating2 ? (
                  <>
                    <Loader2 size={24} className="animate-spin" />
                    Validating faces...
                  </>
                ) : (
                  <>
                    <Scan size={24} />
                    Analyze Kinship
                  </>
                )}
              </span>
              <div className="absolute inset-0 bg-gradient-to-r from-cyber-pink via-cyber-purple to-cyber-blue opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-8 p-4 rounded-xl glass border border-red-500/50 bg-red-500/10">
              <div className="flex items-center gap-3">
                <AlertCircle size={24} className="text-red-500 flex-shrink-0" />
                <p className="text-red-300">{error}</p>
              </div>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="glass rounded-2xl p-8 border border-cyber-blue/30 glow-border animate-float">
              <div className="text-center">
                <div className="mb-6">
                  {result.related ? (
                    <CheckCircle2 size={64} className="text-green-400 mx-auto animate-pulse" />
                  ) : (
                    <AlertCircle size={64} className="text-orange-400 mx-auto animate-pulse" />
                  )}
                </div>

                <h2 className="text-3xl font-bold mb-4 gradient-text">
                  Analysis Complete
                </h2>

                {/* Main Relationship Type Display */}
                {result.relationship_type && (
                  <div className="mb-6">
                    <div className="glass rounded-xl p-6 border-2 border-cyber-blue/40 bg-gradient-to-br from-cyber-blue/10 to-cyber-purple/10">
                      <p className="text-gray-400 text-sm mb-2">Detected Relationship</p>
                      <p className={`text-4xl font-bold mb-2 ${
                        result.related ? 'text-green-400' : 'text-orange-400'
                      }`}>
                        {result.relationship_type}
                      </p>
                      {result.confidence_score && (
                        <p className="text-cyber-blue text-lg">
                          {(result.confidence_score * 100).toFixed(1)}% Confidence
                        </p>
                      )}
                    </div>
                  </div>
                )}

                <div className="grid md:grid-cols-3 gap-6 mb-6">
                  <div className="glass rounded-xl p-6 border border-cyber-blue/20">
                    <p className="text-gray-400 text-sm mb-2">Kinship Score</p>
                    <p className="text-4xl font-bold text-cyber-blue">
                      {(result.kinship_score * 100).toFixed(1)}%
                    </p>
                  </div>

                  <div className="glass rounded-xl p-6 border border-cyber-purple/20">
                    <p className="text-gray-400 text-sm mb-2">Status</p>
                    <p className={`text-2xl font-bold ${
                      result.related ? 'text-green-400' : 'text-orange-400'
                    }`}>
                      {result.related ? 'Related' : 'Not Related'}
                    </p>
                  </div>

                  <div className="glass rounded-xl p-6 border border-cyber-pink/20">
                    <p className="text-gray-400 text-sm mb-2">Confidence</p>
                    <p className="text-2xl font-bold text-cyber-pink">
                      {result.confidence}
                    </p>
                  </div>
                </div>

                {/* Top Predictions (for multi-class model) */}
                {result.top_predictions && result.top_predictions.length > 0 && (
                  <div className="mb-6">
                    <div className="text-left glass rounded-xl p-6 border border-cyber-purple/20">
                      <h3 className="text-lg font-semibold mb-4 text-cyber-purple text-center">
                        Top Predictions
                      </h3>
                      <div className="space-y-3">
                        {result.top_predictions.map((pred, idx) => (
                          <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-black/20">
                            <div className="flex items-center gap-3">
                              <span className={`text-2xl font-bold ${
                                idx === 0 ? 'text-cyber-blue' : 
                                idx === 1 ? 'text-cyber-purple' : 'text-cyber-pink'
                              }`}>
                                #{idx + 1}
                              </span>
                              <span className="text-white font-semibold">{pred.relationship}</span>
                            </div>
                            <div className="flex items-center gap-3">
                              <div className="w-32 bg-gray-700 rounded-full h-2">
                                <div 
                                  className={`h-2 rounded-full ${
                                    idx === 0 ? 'bg-cyber-blue' : 
                                    idx === 1 ? 'bg-cyber-purple' : 'bg-cyber-pink'
                                  }`}
                                  style={{ width: `${pred.percentage}%` }}
                                ></div>
                              </div>
                              <span className="text-white font-mono font-bold w-16 text-right">
                                {pred.percentage}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* All Probabilities (expandable) */}
                {result.all_probabilities && (
                  <div className="mb-6">
                    <div className="text-left glass rounded-xl p-6 border border-cyber-blue/10">
                      <h3 className="text-lg font-semibold mb-3 text-cyber-blue">
                        All Relationship Probabilities
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                        {Object.entries(result.all_probabilities).map(([relationship, prob]) => (
                          <div key={relationship} className="flex justify-between items-center p-2 rounded bg-black/20">
                            <span className="text-gray-300">{relationship}:</span>
                            <span className="text-white font-mono font-bold">
                              {(prob * 100).toFixed(2)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Technical Details */}
                <div className="text-left glass rounded-xl p-6 border border-cyber-blue/10">
                  <h3 className="text-lg font-semibold mb-3 text-cyber-blue">
                    Technical Details
                  </h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Model Type:</span>
                      <span className="ml-2 text-white font-mono">
                        {result.model_type === 'relationship_classifier' ? 'Multi-Class' : 'Binary'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Confidence Score:</span>
                      <span className="ml-2 text-white font-mono">
                        {result.confidence_score || result.kinship_score}
                      </span>
                    </div>
                    {result.distance && (
                      <div>
                        <span className="text-gray-400">Euclidean Distance:</span>
                        <span className="ml-2 text-white font-mono">{result.distance}</span>
                      </div>
                    )}
                  </div>
                </div>

                <p className="text-gray-400 text-sm mt-6">
                  {result.related 
                    ? `✓ The facial features suggest a ${result.relationship_type || 'biological relationship'} between these individuals.`
                    : '✗ The facial features do not indicate a strong biological relationship.'}
                </p>
              </div>
            </div>
          )}

          {/* Info Section */}
          <div className="mt-12 text-center">
            <div className="glass rounded-xl p-6 border border-cyber-blue/10">
              <h3 className="text-xl font-bold mb-3 text-cyber-blue">How It Works</h3>
              <div className="grid md:grid-cols-3 gap-6 text-sm text-gray-300">
                <div>
                  <div className="text-3xl mb-2">🧠</div>
                  <h4 className="font-semibold mb-1">Siamese CNN</h4>
                  <p className="text-xs text-gray-400">
                    Twin neural networks extract facial features from both images
                  </p>
                </div>
                <div>
                  <div className="text-3xl mb-2">📊</div>
                  <h4 className="font-semibold mb-1">Feature Comparison</h4>
                  <p className="text-xs text-gray-400">
                    Computes similarity using contrastive loss and euclidean distance
                  </p>
                </div>
                <div>
                  <div className="text-3xl mb-2">✨</div>
                  <h4 className="font-semibold mb-1">Kinship Score</h4>
                  <p className="text-xs text-gray-400">
                    Generates probability score indicating biological relationship
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 text-gray-500 text-sm">
          <p>Powered by TensorFlow & React | KinFaceW-II Dataset</p>
          <p className="mt-2">Face Kinship Verification System v1.0</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
