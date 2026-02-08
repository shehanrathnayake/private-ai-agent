import requests
import sys
import time
import argparse
from datetime import datetime

def print_banner(host, session_id):
    print("\n" + "="*50)
    print(f"Connected to agent @ {host}")
    print(f"Session: {session_id}")
    print("Type 'exit' or 'quit' to end the session.")
    print("Toggle debug mode with ':debug'")
    print("="*50 + "\n")

def chat():
    parser = argparse.ArgumentParser(description="Terminal Chat Client for Private AI Agent")
    parser.add_argument("host", help="Agent host IP or hostname")
    parser.add_argument("session_id", help="Session ID for the conversation")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (response times)")
    args = parser.parse_args()

    host = args.host
    session_id = args.session_id
    debug_mode = args.debug
    url = f"http://{host}:8000/ask"

    print_banner(host, session_id)

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\n\nExiting chat. Goodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nExiting chat. Goodbye!")
                break
            
            if user_input.lower() == ":debug":
                debug_mode = not debug_mode
                print(f"[*] Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue

            payload = {
                "message": user_input,
                "session_id": session_id
            }

            try:
                start_time = time.time()
                response = requests.post(url, json=payload, timeout=60)
                elapsed_time = time.time() - start_time
                response.raise_for_status()
                
                agent_reply = response.json().get("response", "No response from agent.")
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                print(f"\n[{timestamp}] Agent: {agent_reply}")
                
                if debug_mode:
                    print(f"    [DEBUG] Response time: {elapsed_time:.2f}s")
                print() # Extra newline for spacing

            except requests.exceptions.RequestException as e:
                print(f"\n[!] Error communicating with agent: {e}\n")

    except KeyboardInterrupt:
        print("\n\nSession interrupted by user. Shutting down gracefully...")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")

if __name__ == "__main__":
    chat()
