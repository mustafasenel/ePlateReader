#!/usr/bin/env python3
"""Test script for LLM Vision Service."""

import base64
import requests
import sys
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_health(base_url: str = "http://localhost:8000"):
    """Test health endpoint."""
    print("\n🔍 Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_plate_recognition(image_path: str, base_url: str = "http://localhost:8000"):
    """Test plate recognition endpoint."""
    print(f"\n🚗 Testing plate recognition with: {image_path}")
    
    # Encode image
    image_base64 = encode_image(image_path)
    
    # Send request
    response = requests.post(
        f"{base_url}/api/v1/recognize/plate",
        json={"image_base64": image_base64}
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    
    if result.get("success"):
        print(f"✅ Plate: {result['plate_text']} (Confidence: {result['confidence']:.2%})")
        print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    return result


def test_general_query(image_path: str, prompt: str, base_url: str = "http://localhost:8000"):
    """Test general query endpoint."""
    print(f"\n💬 Testing general query with: {image_path}")
    print(f"Prompt: {prompt}")
    
    # Encode image
    image_base64 = encode_image(image_path)
    
    # Send request
    response = requests.post(
        f"{base_url}/api/v1/query",
        json={
            "image_base64": image_base64,
            "prompt": prompt,
            "max_tokens": 100
        }
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result.get("success"):
        print(f"✅ Response: {result['response']}")
        print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    return result


def main():
    """Run tests."""
    if len(sys.argv) < 2:
        print("Usage: python test_service.py <image_path> [base_url]")
        print("Example: python test_service.py testImages/1.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    print("="*60)
    print("LLM Vision Service - Test Suite")
    print("="*60)
    
    # Test 1: Health check
    if not test_health(base_url):
        print("\n❌ Service is not healthy. Please check if the service is running.")
        sys.exit(1)
    
    # Test 2: Plate recognition
    test_plate_recognition(image_path, base_url)
    
    # Test 3: General query
    test_general_query(
        image_path,
        "Describe what you see in this image in detail.",
        base_url
    )
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
