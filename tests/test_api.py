#!/usr/bin/env python3
"""
Test script for the Classification Service API.

Usage:
    python test_api.py [--base-url http://localhost:8000]
"""

import argparse
import json
import sys
import requests


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("\n1. Testing /api/v1/health...")
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_ready(base_url: str) -> bool:
    """Test readiness endpoint."""
    print("\n2. Testing /api/v1/health/ready...")
    try:
        response = requests.get(f"{base_url}/api/v1/health/ready", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_register(base_url: str, username: str, password: str) -> bool:
    """Test user registration."""
    print(f"\n3. Testing /api/v1/auth/register (user: {username})...")
    try:
        response = requests.post(
            f"{base_url}/api/v1/auth/register",
            json={"username": username, "password": password},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code in [200, 201, 400]  # 400 if user exists
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_login(base_url: str, username: str, password: str) -> str:
    """Test login and return token."""
    print(f"\n4. Testing /api/v1/auth/token (user: {username})...")
    try:
        response = requests.post(
            f"{base_url}/api/v1/auth/token",
            data={"username": username, "password": password},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            token = response.json().get("access_token")
            print(f"   Token received: {token[:20]}..." if token else "   No token")
            return token
        else:
            print(f"   Response: {response.json()}")
            return None
    except Exception as e:
        print(f"   ERROR: {e}")
        return None


def test_classify(base_url: str, token: str) -> bool:
    """Test classification endpoint."""
    print("\n5. Testing /api/v1/classify...")

    request_data = {
        "prompt": "Rate this product review from 1 to 5 stars based on sentiment.",
        "input_data": {
            "text": "This product is amazing! Best purchase I've ever made. Highly recommend to everyone!",
            "title": "Excellent product"
        },
        "options": {
            "return_probabilities": True,
            "return_raw_response": False
        }
    }

    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            f"{base_url}/api/v1/classify",
            json=request_data,
            headers=headers,
            timeout=120  # Classification can take a while
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_classify_unauthorized(base_url: str) -> bool:
    """Test that classification requires auth."""
    print("\n6. Testing /api/v1/classify (no auth - should fail)...")
    try:
        response = requests.post(
            f"{base_url}/api/v1/classify",
            json={
                "prompt": "Test",
                "input_data": {"text": "Test"}
            },
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 401
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_stats(base_url: str, token: str) -> bool:
    """Test system stats endpoint."""
    print("\n7. Testing /api/v1/classify/stats...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{base_url}/api/v1/classify/stats",
            headers=headers,
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Classification Service API")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--username",
        default="testuser",
        help="Test username (default: testuser)"
    )
    parser.add_argument(
        "--password",
        default="testpassword123",
        help="Test password (default: testpassword123)"
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Skip classification test (useful if models not loaded)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Classification Service API Test")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")

    results = []

    # Test health
    results.append(("Health Check", test_health(args.base_url)))

    # Test readiness
    results.append(("Readiness Check", test_ready(args.base_url)))

    # Test registration
    results.append(("User Registration", test_register(
        args.base_url, args.username, args.password
    )))

    # Test login
    token = test_login(args.base_url, args.username, args.password)
    results.append(("User Login", token is not None))

    if token:
        # Test unauthorized access
        results.append(("Auth Required Check", test_classify_unauthorized(args.base_url)))

        # Test stats
        results.append(("System Stats", test_stats(args.base_url, token)))

        # Test classification
        if not args.skip_classify:
            results.append(("Classification", test_classify(args.base_url, token)))
        else:
            print("\n5. Skipping classification test (--skip-classify)")

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
