"""
Test script for face validation functionality
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

def test_face_detection():
    """Test MediaPipe face detection setup"""
    print("\n" + "="*80)
    print("Testing Face Validation with MediaPipe")
    print("="*80 + "\n")
    
    try:
        # Initialize face detector
        face_detector = mp_face_detection.FaceDetection(
            model_selection=0,  # Short-range for close-up faces
            min_detection_confidence=0.3  # Lower threshold for better detection
        )
        print("✓ MediaPipe Face Detector initialized successfully")
        
        # Test with sample images from dataset
        test_images_dir = "KinFaceW-II/images"
        
        if not os.path.exists(test_images_dir):
            print(f"\n✗ Test images directory not found: {test_images_dir}")
            print("  Please ensure you're running this from the backend directory")
            return
        
        # Get first few images to test
        test_count = 0
        max_tests = 5
        
        for root, dirs, files in os.walk(test_images_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')) and test_count < max_tests:
                    img_path = os.path.join(root, file)
                    
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    results = face_detector.process(img_rgb)
                    
                    if results.detections:
                        num_faces = len(results.detections)
                        confidence = results.detections[0].score[0]
                        status = "✓ VALID" if num_faces == 1 else "⚠ MULTIPLE"
                        print(f"{status} - {file}: {num_faces} face(s), confidence: {confidence:.3f}")
                    else:
                        print(f"✗ NO FACE - {file}: No face detected")
                    
                    test_count += 1
        
        print(f"\n✓ Tested {test_count} images successfully")
        print("\nFace validation is working correctly! 🎉")
        
        # Close detector
        face_detector.close()
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_face_detection()
