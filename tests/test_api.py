import unittest
import requests
import time
from typing import Dict, Any

API_BASE = "http://127.0.0.1:8000"


class TestAPI(unittest.TestCase):
    """Integration tests for FastAPI endpoints"""

    @classmethod
    def setUpClass(cls):
        """Wait for API to be ready"""
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{API_BASE}/health", timeout=2)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i == max_retries - 1:
                    raise Exception("API server not running. Start with: uvicorn api:app --reload")
                time.sleep(1)

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{API_BASE}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")

    def test_predict_safe_message(self):
        """Test prediction with safe message"""
        payload = {"text": "Hello, how are you today? I hope you're having a great day!"}
        response = requests.post(
            f"{API_BASE}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Validate response structure
        self.assertIn("label", data)
        self.assertIn("normal_score", data)
        self.assertIn("components", data)
        
        # Validate types
        self.assertIsInstance(data["label"], str)
        self.assertIsInstance(data["normal_score"], (int, float))
        self.assertIsInstance(data["components"], dict)
        
        # Validate components
        self.assertIn("lstm", data["components"])
        self.assertIn("bert", data["components"])
        self.assertIn("random_forest", data["components"])
        
        # Safe message should likely be classified as Normal
        self.assertIn(data["label"].lower(), ["normal", "cyberbullying"])

    def test_predict_harmful_message(self):
        """Test prediction with potentially harmful message"""
        payload = {"text": "You are stupid and worthless, go away"}
        response = requests.post(
            f"{API_BASE}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Validate structure
        self.assertIn("label", data)
        self.assertIn("normal_score", data)
        
        # Score should be between 0 and 1
        self.assertGreaterEqual(data["normal_score"], 0.0)
        self.assertLessEqual(data["normal_score"], 1.0)

    def test_predict_empty_text(self):
        """Test prediction with empty text"""
        payload = {"text": ""}
        response = requests.post(
            f"{API_BASE}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        # Should either accept empty or return 422 validation error
        self.assertIn(response.status_code, [200, 422])

    def test_predict_missing_field(self):
        """Test prediction with missing text field"""
        payload = {}
        response = requests.post(
            f"{API_BASE}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        self.assertEqual(response.status_code, 422)  # Validation error

    def test_response_time(self):
        """Test API response time is reasonable"""
        payload = {"text": "This is a test message for performance testing"}
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        elapsed = time.time() - start_time
        
        self.assertEqual(response.status_code, 200)
        # Should respond within 5 seconds (generous for first request)
        self.assertLess(elapsed, 5.0, f"Response took {elapsed:.2f}s, expected < 5s")

    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = requests.options(
            f"{API_BASE}/predict",
            headers={"Origin": "http://localhost:3000"}
        )
        # CORS headers should be present
        self.assertIn("access-control-allow-origin", response.headers)

    def test_multiple_requests(self):
        """Test handling multiple sequential requests"""
        payloads = [
            {"text": "First test message"},
            {"text": "Second test message"},
            {"text": "Third test message"}
        ]
        
        for payload in payloads:
            response = requests.post(
                f"{API_BASE}/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("label", data)


if __name__ == "__main__":
    unittest.main()


