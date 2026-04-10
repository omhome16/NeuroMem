import requests
import uuid
import json

API_BASE = "http://localhost:8000"
API_KEY = "neuromem-dev-key-change-me"

def test_isolation():
    user_a = f"user_A_{uuid.uuid4().hex[:6]}"
    user_b = f"user_B_{uuid.uuid4().hex[:6]}"
    
    results = {"user_a": user_a, "user_b": user_b}
    
    try:
        # 1. User A sends a message
        res = requests.post(
            f"{API_BASE}/chat",
            headers={"X-API-KEY": API_KEY, "X-USER-ID": user_a},
            json={"message": "My secret code is 1234. Remember this.", "include_memory_debug": True}
        )
        data = res.json()
        working_a = [m for m in data['memory_debug']['memories'] if m['tier'] == 'WORKING']
        results["user_a_working_count"] = len(working_a)
        
        # 2. User B checks state
        res = requests.get(
            f"{API_BASE}/memory/state",
            headers={"X-API-KEY": API_KEY, "X-USER-ID": user_b}
        )
        state_b = res.json()
        working_b = state_b.get('working', [])
        results["user_b_working_count"] = len(working_b)
        
        if len(working_b) == 0 and len(working_a) > 0:
            results["status"] = "SUCCESS"
        else:
            results["status"] = "FAILURE"
            results["details"] = working_b
            
    except Exception as e:
        results["status"] = "ERROR"
        results["error"] = str(e)

    with open("test_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    test_isolation()
