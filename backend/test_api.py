"""
Test script for the Face Kinship Verification API
"""
import requests
import os
import base64
from pathlib import Path

API_URL = 'http://localhost:5000'
DATASET_PATH = r'c:\Users\patil\Desktop\Project xlg\KinFaceW-II\images'

def test_health():
    """Test health endpoint."""
    print("\n" + "="*80)
    print("Testing /health endpoint...")
    print("="*80)
    
    try:
        response = requests.get(f'{API_URL}/health')
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_with_files():
    """Test /analyze endpoint with file upload."""
    print("\n" + "="*80)
    print("Testing /analyze endpoint with file upload...")
    print("="*80)
    
    # Use sample images from dataset
    img1_path = os.path.join(DATASET_PATH, 'father-dau', 'fd_001_1.jpg')
    img2_path = os.path.join(DATASET_PATH, 'father-dau', 'fd_001_2.jpg')
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Error: Sample images not found!")
        print(f"  Image 1: {img1_path}")
        print(f"  Image 2: {img2_path}")
        return False
    
    try:
        with open(img1_path, 'rb') as f1, open(img2_path, 'rb') as f2:
            files = {
                'image1': ('face1.jpg', f1, 'image/jpeg'),
                'image2': ('face2.jpg', f2, 'image/jpeg')
            }
            
            print(f"Uploading images:")
            print(f"  Image 1: {img1_path}")
            print(f"  Image 2: {img2_path}")
            print(f"\nSending request...")
            
            response = requests.post(f'{API_URL}/analyze', files=files)
            
            print(f"\nStatus Code: {response.status_code}")
            print(f"Response:")
            
            result = response.json()
            for key, value in result.items():
                print(f"  {key}: {value}")
            
            return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_with_base64():
    """Test /analyze endpoint with base64 encoded images."""
    print("\n" + "="*80)
    print("Testing /analyze endpoint with base64 encoding...")
    print("="*80)
    
    # Use different sample images
    img1_path = os.path.join(DATASET_PATH, 'father-son', 'fs_001_1.jpg')
    img2_path = os.path.join(DATASET_PATH, 'father-son', 'fs_001_2.jpg')
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Error: Sample images not found!")
        return False
    
    try:
        with open(img1_path, 'rb') as f1, open(img2_path, 'rb') as f2:
            img1_b64 = base64.b64encode(f1.read()).decode()
            img2_b64 = base64.b64encode(f2.read()).decode()
        
        print(f"Encoding images to base64:")
        print(f"  Image 1: {img1_path}")
        print(f"  Image 2: {img2_path}")
        print(f"\nSending request...")
        
        response = requests.post(
            f'{API_URL}/analyze',
            json={
                'image1': img1_b64,
                'image2': img2_b64
            },
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response:")
        
        result = response.json()
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_negative_pair():
    """Test with non-related pair."""
    print("\n" + "="*80)
    print("Testing with non-related pair...")
    print("="*80)
    
    # Use images from different families
    img1_path = os.path.join(DATASET_PATH, 'father-dau', 'fd_001_1.jpg')
    img2_path = os.path.join(DATASET_PATH, 'father-son', 'fs_050_2.jpg')
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Error: Sample images not found!")
        return False
    
    try:
        with open(img1_path, 'rb') as f1, open(img2_path, 'rb') as f2:
            files = {
                'image1': ('face1.jpg', f1, 'image/jpeg'),
                'image2': ('face2.jpg', f2, 'image/jpeg')
            }
            
            print(f"Uploading non-related images:")
            print(f"  Image 1: {img1_path}")
            print(f"  Image 2: {img2_path}")
            print(f"\nSending request...")
            
            response = requests.post(f'{API_URL}/analyze', files=files)
            
            print(f"\nStatus Code: {response.status_code}")
            print(f"Response:")
            
            result = response.json()
            for key, value in result.items():
                print(f"  {key}: {value}")
            
            print(f"\nExpected: Low kinship score (should be Not Related)")
            
            return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_error_handling():
    """Test error handling."""
    print("\n" + "="*80)
    print("Testing error handling...")
    print("="*80)
    
    try:
        # Test with missing image
        print("\n1. Testing with missing image...")
        response = requests.post(f'{API_URL}/analyze', json={'image1': 'test'})
        print(f"   Status Code: {response.status_code} (Expected: 400)")
        print(f"   Response: {response.json()}")
        
        # Test with invalid base64
        print("\n2. Testing with invalid base64...")
        response = requests.post(
            f'{API_URL}/analyze',
            json={'image1': 'invalid', 'image2': 'invalid'}
        )
        print(f"   Status Code: {response.status_code} (Expected: 400)")
        print(f"   Response: {response.json()}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("FACE KINSHIP VERIFICATION API - TEST SUITE")
    print("="*80)
    
    results = {
        'Health Check': test_health(),
        'File Upload Test': test_with_files(),
        'Base64 Test': test_with_base64(),
        'Negative Pair Test': test_negative_pair(),
        'Error Handling Test': test_error_handling()
    }
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80)

if __name__ == "__main__":
    print("\nMake sure the Flask server is running on http://localhost:5000")
    print("Press Enter to start tests...")
    input()
    
    run_all_tests()
