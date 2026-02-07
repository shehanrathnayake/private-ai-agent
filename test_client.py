import requests
import sys

def test_agent(ip, message, session_id="test_user"):
    url = f"http://{ip}:8000/ask"
    payload = {
        "message": message,
        "session_id": session_id
    }
    
    try:
        print(f"Sending message to {url}...")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("\nAgent Response:")
        print("-" * 20)
        print(response.json().get("response"))
        print("-" * 20)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_client.py <AGENT_IP> '<MESSAGE>' [session_id]")
        print("Example: python test_client.py 192.168.1.45 'Hello agent!' my-session-123")
    else:
        session = sys.argv[3] if len(sys.argv) > 3 else "test_user"
        test_agent(sys.argv[1], sys.argv[2], session)
