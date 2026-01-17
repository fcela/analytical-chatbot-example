"""
Launcher script to start both the A2A Backend and the REST API (BFF).
"""
import multiprocessing
import os
import time
import sys

def run_backend():
    print("Starting A2A Backend (gRPC + HTTP)...")
    # Import here to avoid interference
    import uvicorn
    from a2a_server import create_http_app, create_grpc_server
    
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    # Backend HTTP port (internal/debug) - Changed from 8000 to 8001
    http_port = int(os.getenv("BACKEND_HTTP_PORT", "8001")) 
    # Backend gRPC port
    grpc_port = int(os.getenv("BACKEND_GRPC_PORT", "50051"))
    
    # Create HTTP app
    http_app = create_http_app(host, http_port, grpc_port)
    
    # Configure Uvicorn
    config = uvicorn.Config(app=http_app, host=host, port=http_port, log_level="info")
    server = uvicorn.Server(config)
    
    # Run gRPC + HTTP
    import asyncio
    
    async def main():
        tasks = [server.serve()]
        
        print(f"Initializing gRPC server on port {grpc_port}...", flush=True)
        grpc_server = create_grpc_server(host, http_port, grpc_port)
        if grpc_server:
            tasks.append(grpc_server.serve())
            print(f"gRPC server starting on {host}:{grpc_port}", flush=True)
        else:
            print("Failed to create gRPC server", flush=True)
            
        print(f"Backend HTTP starting on {host}:{http_port}", flush=True)
        await asyncio.gather(*tasks)

    asyncio.run(main())

def run_frontend_api():
    print("Starting Frontend REST API (BFF)...")
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    # Frontend API port - Stays 8000 for the UI
    port = int(os.getenv("PORT", "8000"))
    
    # Need to tell REST server where backend is
    os.environ["BACKEND_PORT"] = os.getenv("BACKEND_GRPC_PORT", "50051")
    
    uvicorn.run("rest_server:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    # Create processes
    backend_process = multiprocessing.Process(target=run_backend)
    frontend_process = multiprocessing.Process(target=run_frontend_api)
    
    # Start Backend first
    backend_process.start()
    time.sleep(3) # Give backend a moment to start gRPC
    
    # Start Frontend
    frontend_process.start()
    
    try:
        backend_process.join()
        frontend_process.join()
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.join()
        frontend_process.join()