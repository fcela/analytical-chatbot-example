.PHONY: install start-backend start-backend-legacy start-frontend

install:
	pip install -r requirements.txt
	cd frontend && npm install

# Run the A2A server (default)
start-backend:
	cd .. && python main.py

# Run the legacy FastAPI server (for backwards compatibility)
start-backend-legacy:
	uvicorn api:app --reload --port 8000

start-frontend:
	cd frontend && npm run dev