"""
Face Kinship Verification API
==============================
Flask REST API for kinship verification using deep learning.
"""

import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Import custom layers for model loading
from custom_layers import AbsoluteDifference, L2Distance, CosineSimilarity

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
class Config:
    MODEL_PATH = 'model/kinship_verification_kinfacew2.keras'
    IMG_SIZE = 64
    THRESHOLD = 0.5  # Threshold for kinship classification

config = Config()

# Global model variable
model = None


def load_model():
    """Load the trained Keras model with custom objects."""
    global model
    
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {config.MODEL_PATH}. "
            "Please train the model first using train_kinfacew2.py"
        )
    
    try:
        # Load model with custom objects
        custom_objects = {
            'AbsoluteDifference': AbsoluteDifference,
            'L2Distance': L2Distance,
            'CosineSimilarity': CosineSimilarity
        }
        
        model = keras.models.load_model(
            config.MODEL_PATH,
            custom_objects=custom_objects
        )
        
        print(f"✓ Model loaded successfully from {config.MODEL_PATH}")
        try:
            input_names = [inp.name for inp in model.inputs] if hasattr(model, 'inputs') else 'N/A'
            print(f"  Model inputs: {input_names}")
        except:
            print(f"  Model loaded (inputs: 2 images)")
        print(f"  Model ready for predictions!")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


def preprocess_image(image_data, source='file'):
    """
    Preprocess image for model input.
    
    Args:
        image_data: Image data (file object or base64 string)
        source: 'file' or 'base64'
    
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image
        if source == 'base64':
            # Decode base64
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))
            img = np.array(img)
        else:
            # Read from file object
            img_bytes = image_data.read()
            img = Image.open(io.BytesIO(img_bytes))
            img = np.array(img)
        
        # Convert to RGB if needed
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize to model input size
        img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def predict_kinship(img1, img2):
    """
    Predict kinship between two face images.
    
    Args:
        img1: First preprocessed image
        img2: Second preprocessed image
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Add batch dimension
        img1_batch = np.expand_dims(img1, axis=0)
        img2_batch = np.expand_dims(img2, axis=0)
        
        # Make prediction
        prediction = model.predict([img1_batch, img2_batch], verbose=0)
        kinship_score = float(prediction[0][0])
        
        # Determine if related
        is_related = kinship_score >= config.THRESHOLD
        
        # Calculate confidence
        if is_related:
            confidence = kinship_score
            confidence_label = "High" if confidence > 0.7 else "Medium"
        else:
            confidence = 1 - kinship_score
            confidence_label = "High" if confidence > 0.7 else "Medium"
        
        # Prepare result
        result = {
            'kinship_score': round(kinship_score, 4),
            'related': bool(is_related),
            'confidence': confidence_label,
            'confidence_score': round(confidence, 4),
            'threshold': config.THRESHOLD,
            'model_type': 'binary_classifier',
            'relationship_type': 'Biological Relatives' if is_related else 'Not Related'
        }
        
        return result
    
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': config.MODEL_PATH,
        'version': '1.0.0'
    })


@app.route('/analyze', methods=['POST'])
def analyze_kinship():
    """
    Main endpoint for kinship verification.
    Accepts either file uploads or base64 encoded images.
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please restart the server.'
            }), 500
        
        # Handle file uploads
        if 'image1' in request.files and 'image2' in request.files:
            img1_file = request.files['image1']
            img2_file = request.files['image2']
            
            # Preprocess images
            img1 = preprocess_image(img1_file, source='file')
            img2 = preprocess_image(img2_file, source='file')
        
        # Handle base64 encoded images
        elif request.is_json:
            data = request.get_json()
            
            if 'image1' not in data or 'image2' not in data:
                return jsonify({
                    'error': 'Both image1 and image2 are required'
                }), 400
            
            # Preprocess images
            img1 = preprocess_image(data['image1'], source='base64')
            img2 = preprocess_image(data['image2'], source='base64')
        
        else:
            return jsonify({
                'error': 'Invalid request format. Send either files or JSON with base64 images.'
            }), 400
        
        # Make prediction
        result = predict_kinship(img1, img2)
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error in analyze_kinship: {e}")
        return jsonify({
            'error': 'Internal server error during analysis',
            'details': str(e)
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        return jsonify({
            'model_name': 'Siamese CNN for Kinship Verification',
            'architecture': 'Siamese Network with Custom Layers',
            'input_shape': [config.IMG_SIZE, config.IMG_SIZE, 3],
            'output_shape': model.output.shape.as_list(),
            'total_parameters': int(model.count_params()),
            'custom_layers': ['AbsoluteDifference', 'L2Distance', 'CosineSimilarity'],
            'threshold': config.THRESHOLD
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Main function to start the Flask server."""
    print("\n" + "="*80)
    print("Face Kinship Verification API")
    print("="*80 + "\n")
    
    # Load model
    print("Loading model...")
    try:
        load_model()
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print("\nPlease train the model first by running:")
        print("  python train_kinfacew2.py")
        return
    
    print("\n" + "="*80)
    print("Server Configuration:")
    print("="*80)
    print(f"  Host: 0.0.0.0")
    print(f"  Port: 5000")
    print(f"  Model: {config.MODEL_PATH}")
    print(f"  Image Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"  Threshold: {config.THRESHOLD}")
    print("\nEndpoints:")
    print("  GET  /health      - Health check")
    print("  POST /analyze     - Analyze kinship between two faces")
    print("  GET  /model-info  - Get model information")
    print("\n" + "="*80)
    print("Server is ready! 🚀")
    print("="*80 + "\n")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
