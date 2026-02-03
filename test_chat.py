
import requests
import time
import subprocess
import sys
import json

def test_chat():
    print("Starting main.py...")
    proc = subprocess.Popen([sys.executable, "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        print("Waiting 10s for startup...")
        time.sleep(10)
        
        url = "http://localhost:8000/chat"
        payload = {"message": "Hello"}
        headers = {"Content-Type": "application/json"}
        
        print(f"Sending POST to {url}...")
        try:
            # Emulate browser: include Origin
            headers["Origin"] = "http://localhost:5173"
            
            # Use session for cookies
            s = requests.Session()
            response = s.post(url, json=payload, headers=headers, timeout=10)
            
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                print("PASS: Chat request succeeded.")
            else:
                print("FAIL: Chat request returned error.")
                
        except Exception as e:
            print(f"FAIL: Request raised exception: {e}")

    finally:
        print("Terminating...")
        proc.terminate()
        try:
            outs, errs = proc.communicate(timeout=5)
            print("\n--- STDOUT ---")
            print(outs)
            print("\n--- STDERR ---")
            print(errs)
        except:
            proc.kill()
            
if __name__ == "__main__":
    test_chat()
