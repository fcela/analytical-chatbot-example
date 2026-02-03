
import requests
import time
import subprocess
import os
import signal
import sys

def check_server(url, name):
    try:
        print(f"Checking {name} at {url}...")
        response = requests.get(url, timeout=2)
        print(f"  {name} Status: {response.status_code}")
        print(f"  {name} Response: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"  {name} is NOT reachable.")
        return False
    except Exception as e:
        print(f"  {name} error: {e}")
        return False

def check_cors(url, origin):
    print(f"Checking CORS for {url} with origin {origin}...")
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    try:
        response = requests.options(url, headers=headers, timeout=2)
        print(f"  Status: {response.status_code}")
        print("  Headers:", response.headers)
        
        aca_origin = response.headers.get("access-control-allow-origin")
        aca_creds = response.headers.get("access-control-allow-credentials")
        
        if aca_origin == origin and aca_creds == 'true':
            print("  PASS: CORS configured correctly.")
        else:
            print(f"  FAIL: CORS headers mismatch. Origin: {aca_origin}, Creds: {aca_creds}")
    except Exception as e:
        print(f"  CORS check error: {e}")

def main():
    # Start the servers in the background
    print("Starting main.py in background...")
    proc = subprocess.Popen([sys.executable, "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        # Wait for startup
        print("Waiting 10 seconds for startup...")
        time.sleep(10)
        
        # Check BFF (REST API)
        bff_up = check_server("http://localhost:8000/", "BFF REST API")
        
        # Check Backend (HTTP part)
        backend_up = check_server("http://localhost:8001/health", "Backend HTTP") # Assuming standard health endpoint or root
        
        # Check gRPC port (basic TCP check)
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 50051))
        if result == 0:
            print("  gRPC Port 50051 is OPEN.")
        else:
            print("  gRPC Port 50051 is CLOSED.")
        sock.close()

        if bff_up:
            check_cors("http://localhost:8000/chat", "http://localhost:5173")

    finally:
        print("Terminating servers...")
        proc.terminate()
        try:
            outs, errs = proc.communicate(timeout=5)
            print("\n--- Server STDOUT ---")
            print(outs if outs else "None")
            print("\n--- Server STDERR ---")
            print(errs if errs else "None")
        except:
            proc.kill()
            proc.wait()

if __name__ == "__main__":
    main()
