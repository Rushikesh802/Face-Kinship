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
import mediapipe as mp

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

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = None


def initialize_face_detector():
    """Initialize MediaPipe face detector."""
    global face_detector
    try:
        face_detector = mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range detection (better for close-up faces)
            min_detection_confidence=0.3  # Lower threshold for better detection
        )
        print("✓ MediaPipe Face Detector initialized")
        return True
    except Exception as e:
        print(f"✗ Error initializing face detector: {e}")
        raise


def validate_face(image_data, source='file', image_name='Image'):
    """
    Validate that the image contains exactly one clear face.
    
    Args:
        image_data: Image data (file object or base64 string)
        source: 'file' or 'base64'
        image_name: Name for error messages (e.g., 'Image 1', 'Image 2')
    
    Returns:
        tuple: (is_valid, error_message, image_array)
    """
    try:
        # Load image
        if source == 'base64':
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)
        else:
            img_bytes = image_data.read()
            # Reset file pointer for later use
            image_data.seek(0)
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img_array.copy()
        
        # Detect faces using MediaPipe
        results = face_detector.process(img_rgb)
        
        # Check if any faces detected
        if not results.detections:
            return False, f"No face detected in {image_name}. Please upload a clear face image.", None
        
        # Check for multiple faces
        num_faces = len(results.detections)
        if num_faces > 1:
            return False, f"Multiple faces detected in {image_name}. Please upload an image with only one face.", None
        
        # Check detection confidence
        detection = results.detections[0]
        confidence = detection.score[0]
        
        if confidence < 0.3:
            return False, f"Face quality too low in {image_name}. Please upload a clearer image.", None
        
        # Face validation passed
        return True, None, img_array
    
    except Exception as e:
        return False, f"Error validating {image_name}: {str(e)}", None


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
        'face_detector_loaded': face_detector is not None,
        'model_path': config.MODEL_PATH,
        'face_validation': 'MediaPipe (confidence >= 0.3)',
        'version': '1.1.0'
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
        
        # Check if face detector is initialized
        if face_detector is None:
            return jsonify({
                'error': 'Face detector not initialized. Please restart the server.'
            }), 500
        
        # Handle file uploads
        if 'image1' in request.files and 'image2' in request.files:
            img1_file = request.files['image1']
            img2_file = request.files['image2']
            
            # Validate Image 1
            is_valid1, error_msg1, img1_array = validate_face(img1_file, source='file', image_name='Image 1')
            if not is_valid1:
                return jsonify({
                    'error': error_msg1,
                    'validation_failed': True,
                    'failed_image': 'image1'
                }), 400
            
            # Validate Image 2
            is_valid2, error_msg2, img2_array = validate_face(img2_file, source='file', image_name='Image 2')
            if not is_valid2:
                return jsonify({
                    'error': error_msg2,
                    'validation_failed': True,
                    'failed_image': 'image2'
                }), 400
            
            # Preprocess validated images
            img1 = preprocess_image(img1_file, source='file')
            img2 = preprocess_image(img2_file, source='file')
        
        # Handle base64 encoded images
        elif request.is_json:
            data = request.get_json()
            
            if 'image1' not in data or 'image2' not in data:
                return jsonify({
                    'error': 'Both image1 and image2 are required'
                }), 400
            
            # Validate Image 1
            is_valid1, error_msg1, img1_array = validate_face(data['image1'], source='base64', image_name='Image 1')
            if not is_valid1:
                return jsonify({
                    'error': error_msg1,
                    'validation_failed': True,
                    'failed_image': 'image1'
                }), 400
            
            # Validate Image 2
            is_valid2, error_msg2, img2_array = validate_face(data['image2'], source='base64', image_name='Image 2')
            if not is_valid2:
                return jsonify({
                    'error': error_msg2,
                    'validation_failed': True,
                    'failed_image': 'image2'
                }), 400
            
            # Preprocess validated images
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


@app.route('/validate', methods=['POST'])
def validate_image():
    """
    Validate a single image for face detection.
    Returns immediately with validation result.
    """
    try:
        # Check if face detector is initialized
        if face_detector is None:
            return jsonify({
                'error': 'Face detector not initialized. Please restart the server.'
            }), 500
        
        # Handle file upload
        if 'image' in request.files:
            img_file = request.files['image']
            
            # Validate face
            is_valid, error_msg, img_array = validate_face(img_file, source='file', image_name='Image')
            
            if not is_valid:
                return jsonify({
                    'valid': False,
                    'error': error_msg
                }), 200  # Return 200 with validation result
            
            return jsonify({
                'valid': True,
                'message': 'Face detected successfully'
            }), 200
        
        # Handle base64 encoded image
        elif request.is_json:
            data = request.get_json()
            
            if 'image' not in data:
                return jsonify({
                    'valid': False,
                    'error': 'Image data is required'
                }), 400
            
            # Validate face
            is_valid, error_msg, img_array = validate_face(data['image'], source='base64', image_name='Image')
            
            if not is_valid:
                return jsonify({
                    'valid': False,
                    'error': error_msg
                }), 200
            
            return jsonify({
                'valid': True,
                'message': 'Face detected successfully'
            }), 200
        
        else:
            return jsonify({
                'valid': False,
                'error': 'Invalid request format. Send either file or JSON with base64 image.'
            }), 400
    
    except Exception as e:
        print(f"Error in validate_image: {e}")
        return jsonify({
            'valid': False,
            'error': f'Validation error: {str(e)}'
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
    
    # Initialize face detector
    print("Initializing face detector...")
    try:
        initialize_face_detector()
    except Exception as e:
        print(f"\n✗ Failed to initialize face detector: {e}")
        print("\nPlease install mediapipe:")
        print("  pip install mediapipe==0.10.9")
        return
    
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
    print(f"  Face Detection: MediaPipe (confidence >= 0.3)")
    print("\nEndpoints:")
    print("  GET  /health      - Health check")
    print("  POST /validate    - Validate single image for face detection")
    print("  POST /analyze     - Analyze kinship between two faces")
    print("  GET  /model-info  - Get model information")
    print("\n" + "="*80)
    print("Server is ready! 🚀")
    print("="*80 + "\n")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
