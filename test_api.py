"""
Quick test script to verify API endpoints work correctly.
Run this after starting the API server with: uvicorn app.main:app --port 8000

Usage:
    python test_api.py
"""

import requests
import json
import base64
from PIL import Image
import io

BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("🧪 Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Status: {data['status']}")
    print(f"✅ Endpoints: {data['endpoints']}")
    print()

def test_text_generation():
    """Test text generation endpoint"""
    print("🧪 Testing text generation...")
    payload = {"start_word": "The", "length": 10}
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Generated text: {data['generated_text']}")
    print()

def test_embedding():
    """Test embedding endpoint"""
    print("🧪 Testing word embedding...")
    payload = {"word": "hello"}
    response = requests.post(f"{BASE_URL}/embed", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Word: {data['word']}, Dimension: {data['dim']}")
    print()

def test_gan_generate():
    """Test GAN generation endpoint"""
    print("🧪 Testing GAN generation...")
    response = requests.get(f"{BASE_URL}/gan/generate?num_samples=4")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated {data['num_samples']} samples")
        # Decode and display image
        img_data = base64.b64decode(data['image_base64_png'])
        img = Image.open(io.BytesIO(img_data))
        print(f"   Image size: {img.size}")
        img.save("test_gan_output.png")
        print(f"   Saved to: test_gan_output.png")
    elif response.status_code == 500:
        print("⚠️  GAN weights not found - train model first with: python train_gan_offline.py")
    print()

def test_ebm_generate():
    """Test EBM generation endpoint"""
    print("🧪 Testing EBM generation...")
    response = requests.get(f"{BASE_URL}/ebm/generate?num_samples=4&steps=100")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated {data['num_samples']} samples with EBM")
        # Decode and display image
        img_data = base64.b64decode(data['image_base64_png'])
        img = Image.open(io.BytesIO(img_data))
        print(f"   Image size: {img.size}")
        img.save("test_ebm_output.png")
        print(f"   Saved to: test_ebm_output.png")
    elif response.status_code == 500:
        print("⚠️  EBM weights not found - train model first with: python train_ebm_offline.py")
    print()

def test_diffusion_generate():
    """Test Diffusion generation endpoint"""
    print("🧪 Testing Diffusion generation...")
    response = requests.get(f"{BASE_URL}/diffusion/generate?num_samples=4&diffusion_steps=20")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated {data['num_samples']} samples with Diffusion")
        # Decode and display image
        img_data = base64.b64decode(data['image_base64_png'])
        img = Image.open(io.BytesIO(img_data))
        print(f"   Image size: {img.size}")
        img.save("test_diffusion_output.png")
        print(f"   Saved to: test_diffusion_output.png")
    elif response.status_code == 500:
        print("⚠️  Diffusion weights not found - train model first with: python train_diffusion_offline.py")
    print()

def main():
    print("=" * 60)
    print("   API Endpoint Tests")
    print("=" * 60)
    print()
    
    try:
        test_root()
        test_text_generation()
        test_embedding()
        test_gan_generate()
        test_ebm_generate()
        test_diffusion_generate()
        
        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("   Make sure the server is running:")
        print("   uvicorn app.main:app --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

