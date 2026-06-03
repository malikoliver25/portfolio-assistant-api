import unittest
import requests
import time

BASE_URL = "http://127.0.0.1:8000"

class TestPortfolioAgentAPI(unittest.TestCase):

    def test_01_general_chit_chat_routing(self):
        """Assert that non-resume queries route to general chat and return the greeting."""
        payload = {"message": "Hello! Who are you?"}
        
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["meta"]["intent_routed"], "general_chit_chat")
        self.assertEqual(data["meta"]["context_chunks_retrieved"], 0)
        self.assertIn("Malik's AI Assistant", data["response"])

    def test_02_resume_query_routing_and_retrieval(self):
        """Assert that technical resume questions pull the accurate vector chunks."""
        payload = {"message": "Tell me about Malik's experience with Python, FastAPI, and backend development."}
        
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["meta"]["intent_routed"], "portfolio_query")
        self.assertGreaterEqual(data["meta"]["context_chunks_retrieved"], 1)
        
        # Verify the semantic engine surfaced the correct backend experience block
        self.assertIn("Backend Engineering and Data Systems Skills", data["response"])
        self.assertIn("FastAPI", data["response"])

    def test_03_performance_latency_boundary(self):
        """Assert that the complete execution loop runs within high-performance budgets (< 50ms)."""
        payload = {"message": "What AI engineering frameworks does he use?"}
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        end_time = time.time()
        
        # Calculate real-world wall-clock latency
        client_latency_ms = (end_time - start_time) * 1000
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        print(f"\n[PERFORMANCE] Internal API Latency: {data['meta']['performance_ms']}ms")
        print(f"[PERFORMANCE] Total Client Roundtrip: {round(client_latency_ms, 2)}ms")
        
        # Enforce strict performance guardrails
        self.assertLess(data["meta"]["performance_ms"], 50.0, "API internal execution exceeded the 50ms performance budget!")

if __name__ == "__main__":
    print("Launching automated route validation and retrieval assertions...")
    unittest.main()